import os
import json
from pathlib import Path

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from .llama import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download

from mmdet.models import DETECTORS, build_loss 
from mmcv.runner import BaseModule, auto_fp16
from torch.cuda.amp import autocast

@DETECTORS.register_module()
class LLaMA_adapter(nn.Module):

    def __init__(self, llama_ckpt_dir, llama_tokenizer,
                 max_seq_len=512, max_batch_size=32,
                 clip_model='ViT-L/14',
                 v_embed_dim=768, v_depth=8,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=10, query_layer=31,
                 w_bias=False, 
                 w_lora=False, lora_rank=16, 
                 w_new_gate=False,
                 phase="finetune",
                 train_cfg=None,
                 test_cfg=None,
                 quat='fp16',
                 out_layer=15, # unused
                 clip_direct=False,
                 ):
        super().__init__()

        # load llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        # w_bias = phase == "finetune" # TODO
        w_bias = True
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        ) # max_batch_size only affects inferenc

        # 1. clip and clip projector
        self.clip, self.clip_transform = clip.load(clip_model)

        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = query_layer

        self.clip_direct = clip_direct # dirctly use clip visual features

        # 2. visual query, blocks and projector
        self.visual_query = nn.Embedding(query_len, v_embed_dim)
        self.visual_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)])
        self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
        self.visual_proj_norm = nn.LayerNorm(model_args.dim)

        # 3. adapter query
        self.adapter_query = nn.Embedding(
            query_len * query_layer, model_args.dim)

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 5. llama
        model_args.w_bias = w_bias
        model_args.w_lora = w_lora
        model_args.lora_rank = lora_rank
        model_args.w_new_gate = w_new_gate
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_tensor_type(torch.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)

        del self.clip.transformer

         # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, 
                                                    # label_smoothing=0.15
                                                    )

        # 7. training parameters
        self.phase = phase
        self.get_trainable_params(self.phase)

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #        print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True
                    # else:
                    #     para.data = para.data.half()
                    #     para.requires_grad = False

        elif phase == 'pretrain':
            train_param_name = ['gate', 'clip_proj', 'clip_proj_norm', 'visual_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query']
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        elif phase == 'freeze':
            pass
            # for name, para in self.named_parameters():
            #     # para.requires_grad = False
            #     if name.startswith("llama."):
            #         if 'norm' in name or 'bias' in name:
            #             para.data = para.data.float()
        else:
            raise ValueError(f"Unknown model phase: {phase}")
        
    def clip_parallel(self, assist_device):
        self.clip = self.clip.to(assist_device)
        self.clip_proj_norm = self.clip_proj_norm.to(assist_device)
        self.clip_proj = self.clip_proj.to(assist_device)
        self.visual_query = self.visual_query.to(assist_device)
        self.visual_blocks = self.visual_blocks.to(assist_device)
        self.visual_proj = self.visual_proj.to(assist_device)
        self.visual_proj_norm = self.visual_proj_norm.to(assist_device)

    def clip_encode_image(self, x):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.device).to(x.dtype) \
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
            # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x

    def forward_visual(self, imgs):
        clip_feats = []
        for i in range(imgs.shape[1]):
            img = imgs[:, i]
            clip_feat = self.clip_encode_image(img)
            clip_feats.append(clip_feat)
        clip_feats = torch.cat(clip_feats, dim=1)
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

        visual_query = self.visual_query.weight.unsqueeze(
            0).repeat(len(imgs), 1, 1)
        visual_query = torch.cat([visual_query, clip_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)

        visual_query = visual_query[:, :self.query_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query

    # @auto_fp16
    @autocast()
    def forward(self, **kwargs):
        if not self.training:
            return self.generate(**kwargs)
        has_text_label = kwargs['has_text_label']
        # tokens, labels, imgs = kwargs['text_token'][has_text_label], \
        #                        kwargs['text_labels'][has_text_label], \
        #                        kwargs['text_imgs'][has_text_label]
        device = self.visual_query.weight.device
        dtype = self.visual_query.weight.dtype
        tokens, labels, imgs = kwargs['text_token'].to(device), \
                               kwargs['text_labels'].to(device), \
                               kwargs['text_imgs'].to(device)
        visual_query = self.forward_visual(imgs)
        if visual_query.isnan().sum() > 0:
            print(f"visual_query: {visual_query}")
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
        hs = []
        # print('llama layer num:', len(self.llama.layers)) # 32
        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)
            hs.append(h.clone()[:,None,:,:])
        if h.isnan().sum() > 0:
            print(f"h: {h}")
        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1
            hs.append(h.clone()[:,None,:,:])
            if h.isnan().sum() > 0:
                print(f"h@{adapter_index}: {h}")
        hs = torch.cat(hs, dim=1)
        h = self.llama.norm(h)
        output = self.llama.output(h).float() # TODO, has to cast to float here
        ce_output = output[:, :-1, :][has_text_label,]
        ce_labels = labels[:, 1:][has_text_label,]
        if not ce_output.isnan().sum() == 0:
            print(f"ce_output: {ce_output}")
        if not output.isnan().sum() == 0:
            print(f"output: {output}, labels: {labels}")
        if ce_labels.sum() == 0:
            print(f"ce_labels: {ce_labels}")
            output = torch.nan_to_num(output)
            c_loss = output.mean() * 0.0
            # c_loss = torch.tensor(0.0, dtype=output.dtype, device=output.device)
        else:
            # output = torch.nan_to_num(output, nan=0.0, posinf=1e5, neginf=-1e5)
            assert self.llama.vocab_size == 32000
            c_loss = self.criterion(ce_output.reshape(-1, self.llama.vocab_size), 
                                    ce_labels.flatten()
                                    # self.label_smoothing(ce_labels.flatten().long(), eps=0.2)
                                    )
        assert output.isnan().sum() == 0, f"output: {output}"
        assert c_loss.isnan().sum() == 0, f"c_loss: {c_loss}"
        # print('loss:', c_loss, c_loss.type())
        lm_emb = hs[:,:,:-1]
        lm_emb_valid = labels[:,1:]>0
        if self.clip_direct:
            lm_emb = visual_query.unsqueeze(1)
            lm_emb_valid = torch.ones_like(visual_query[:,:,0], device=visual_query.device).bool()

        return {"lm_loss":c_loss}, \
                {
                'lm_emb': lm_emb,
                'lm_emb_valid': lm_emb_valid,
                }

    @torch.inference_mode()
    def forward_inference(self, visual_query, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float('-inf'), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        hs = []
        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)
            hs.append(h.clone()[:,None,-1:,:])
        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1
            hs.append(h.clone()[:,None,-1:,:])
        hs = torch.cat(hs, dim=1)
        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float(),  hs

    @torch.inference_mode()
    def generate(
        self, text_imgs, prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
        **kwargs,
    ):
        clip_device = self.clip_proj.weight.device
        llama_device = self.adapter_query.weight.device
        imgs = text_imgs
        bsz = len(imgs)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)

        with torch.cuda.amp.autocast():
            visual_query = self.forward_visual(imgs.to(clip_device))
            visual_query = visual_query.to(llama_device)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).to(llama_device).long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).to(llama_device).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        hidden_states = []
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits, hidden_state = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            hidden_states.append(hidden_state)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        hidden_states = torch.cat(hidden_states, dim=-2)
        valid_mask = torch.ones_like(hidden_states[:, 0, :, 0], 
                                    dtype=torch.bool, 
                                    device=hidden_states.device)
        return {}, {'decoded':decoded,
                    'lm_emb': hidden_states,
                    'lm_emb_valid': valid_mask,
                    }

    # def init_weights(self):
    def load_model(self, path, map_location='cpu'):
        checkpoint = torch.load(path, map_location=map_location)
        new_checkpoint = {}
        for key, value in checkpoint['model'].items():
            key = key.replace("llma", "llama")
            new_checkpoint[key] = value
        print(self.load_state_dict(new_checkpoint, strict=False))
        import time
        print('loaded llama adapter model:', path)
        time.sleep(3)

_MODELS = {
    "BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    "LORA-BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth",
    "CAPTION-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/5088aeb63a89746b90bcfd5cb819e1c7411b2771b267c6d131ce73e250a8abf0_CAPTION-7B.pth",
    "LORA-BIAS-7B-v21": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.1.0/427dbc27bf62a3ef7a24ffd3ed2c3162_LORA-BIAS-7B-v21.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}

def available_models():
    return list(_MODELS.keys())

def load(name, llama_dir, llama_type="7B", device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts', max_seq_len=512,
        phase="finetune"):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}"), None

    # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
    # llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})

    model = LLaMA_adapter(
        llama_ckpt_dir, llama_tokenzier_path,
        max_seq_len=512, max_batch_size=32,
        clip_model='ViT-L/14',
        v_embed_dim=768, v_depth=8,
        v_num_heads=16, v_mlp_ratio=4.0,
        query_len=10, query_layer=31,
        w_bias=model_cfg.get('w_bias', False), 
        w_lora=model_cfg.get('w_lora', False), 
        lora_rank=model_cfg.get('lora_rank', 16),
        w_new_gate=model_cfg.get('w_lora', False), # for compatibility
        phase=phase)

    load_result = model.load_state_dict(ckpt['model'], strict=False)

    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device), model.clip_transform
