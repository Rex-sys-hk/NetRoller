import time
import copy
import torch
from torch import nn
from torch.cuda.amp import autocast

from mmdet.models import NECKS
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
# from timm.models.layers import Mlp
from .mlp import Mlp

import math

# from projects.mmdet3d_plugin.rollnet.native_sparse_attention_pytorch.native_sparse_attention_pytorch.native_sparse_attention import SparseAttention

@NECKS.register_module()
class RollNetAdaptorBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def init_weights(self):
        for param in self.parameters():
            param.to(torch.float16)

    def forward(self, **kwargs):
        return {}, {}

@NECKS.register_module()
class BEVAdaptor(RollNetAdaptorBase):
    def __init__(self, **kwargs):
        super().__init__()
        # self.model = build_model(kwargs['model'])
        h, w = kwargs['bev_h'], kwargs['bev_w']
        in_dim = kwargs['in_dim']
        out_dim = kwargs['out_dim']
        nhead = kwargs['nhead']
        # self._h, self._w = h//20, w//20
        # self.embedding = nn.Embedding(self._h*self._w, in_dim)
        self.nonsense_embedding = nn.Parameter(torch.randn(1, in_dim), requires_grad=True)
        self.query = nn.Parameter(torch.randn(1, in_dim), requires_grad=True)
        self.CA = nn.MultiheadAttention(in_dim, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.emb_projector = nn.Linear(in_dim, out_dim)
    
    @autocast()
    def forward(self, **kwargs):
        tgt = self.query
        tgt = tgt.unsqueeze(0).repeat(kwargs['lm_emb'].shape[0], 1, 1)
        mem = kwargs['lm_emb']
        if not len(kwargs['lm_emb'].shape) == 3:
            mem = kwargs['lm_emb'][:, -1] # use last lm_emb
        msk = ~kwargs['lm_emb_valid']
        mem = torch.cat((mem, self.nonsense_embedding.repeat(mem.shape[0], 1, 1)), dim=1)
        msk = torch.cat((msk, torch.zeros(msk.shape[0], 1).bool().to(msk.device)), dim=1)
        output, attn = self.CA(tgt, mem, mem,
                        key_padding_mask=msk,
                        )
        output = self.layer_norm(output)
        output = self.emb_projector(output)
        k = min(10, mem.shape[1])
        val, ind = torch.topk(attn[0,-1], k=k)
        labels = kwargs['text_labels'][:,1:]
        labels = torch.cat((labels, -torch.ones(labels.shape[0], 1).long().to(labels.device)), dim=1)
        # print('attn vit val>>:',  val)
        # print('attn vit ind>>:',  ind)
        # print('attn vit lab>>:',  labels[0, ind])
        # print('lm_emb mmm >>:', output.mean().item(), output.max().item(), output.min().item())
        return {}, {'bev_query_bias': output, 'bev_attn': attn}


@NECKS.register_module()
class BEVGridWiseAdaptor(RollNetAdaptorBase):
    def __init__(self, **kwargs):
        super().__init__()
        h, w = kwargs['bev_h'], kwargs['bev_w']
        in_dim = kwargs['in_dim']
        out_dim = kwargs['out_dim']
        nhead = kwargs['nhead']
        self._h, self._w = h, w
        self.pos_embedding = nn.Embedding(self._h*self._w, out_dim)

        self.emb_inprojector = nn.Linear(in_dim, out_dim)
        self.nonsense_embedding = nn.Parameter(torch.randn(1, out_dim), requires_grad=True)

        self.CA = nn.MultiheadAttention(out_dim, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.query_projector = nn.Linear(out_dim, out_dim)
        # self.emb_projector = nn.Linear(out_dim, out_dim)


    def init_weights(self):
        for param in self.parameters():
            param.to(torch.float16)
    
    @autocast()
    def forward(self, **kwargs):
        assert len(kwargs['lm_emb'].shape) == 3
        tgt = self.pos_embedding.weight
        tgt = tgt.unsqueeze(0).repeat(kwargs['lm_emb'].shape[0], 1, 1)

        valid_mems = kwargs['lm_emb_valid'].clone()
        valid_mems = torch.max(valid_mems, dim=0, keepdim=True)[0]
        lm_emb = kwargs['lm_emb'][:, valid_mems[0].bool()]
        msk = ~kwargs['lm_emb_valid'][:, valid_mems[0].bool()]
        mem = self.emb_inprojector(lm_emb)
        mem = torch.cat((mem, self.nonsense_embedding.repeat(mem.shape[0], 1, 1)), dim=1)
        msk = torch.cat((msk, torch.zeros(msk.shape[0], 1).bool().to(msk.device)), dim=1)
        # get bev bias
        output, attn = self.CA(tgt, mem, mem,
                        key_padding_mask=msk,
                        )
        output = self.layer_norm(output)
        bev_query_bias = self.query_projector(output)
        # bev_emb_bias = self.emb_projector(output)
        # get ego query bias
        # tgt = self.ego_query_embedding.weight
        # tgt = tgt.unsqueeze(0).repeat(kwargs['lm_emb'].shape[0], 1, 1)
        # output, attn = self.ego_plan_CA(tgt, mem, mem,
        #                 key_padding_mask=msk,
        #                 )
        # output = self.ego_query_layer_norm(output)
        # ego_query_bias = self.ego_query_projector(output)

        # visualize
        # k = min(10, mem.shape[1])
        # val, ind = torch.topk(attn[0,-1], k=k)
        # labels = kwargs['text_labels'][:,1:]
        # labels = torch.cat((labels, -torch.ones(labels.shape[0], 1).long().to(labels.device)), dim=1)
        # print('attn vit val>>:',  val)
        # print('attn vit ind>>:',  ind)
        # print('attn vit lab>>:',  labels[0, ind])
        # print('lm_emb mmm >>:', output.mean().item(), output.max().item(), output.min().item())
        return {}, {'bev_query_bias': bev_query_bias, 
                    # 'bev_emb_bias': bev_emb_bias,
                    # 'ego_query_bias': ego_query_bias,
                    'bev_attn': attn}

@NECKS.register_module()
class BEVGridWiseAdaptorV2(RollNetAdaptorBase):
    def __init__(self, **kwargs):
        super().__init__()
        h, w = kwargs['bev_h'], kwargs['bev_w']
        in_dim = kwargs['in_dim']
        out_dim = kwargs['out_dim']
        nhead = kwargs['nhead']
        self._h, self._w = h, w
        self.pos_embedding = nn.Embedding(self._h*self._w, 2*out_dim)
        # self.emb_projector = Mlp(in_dim, out_dim*4, out_dim)
        self.emb_inprojector = nn.Linear(in_dim, 2*out_dim)
        # self.emb_inprojector = Mlp(in_dim, out_dim*4, out_dim)
        self.nonsense_embedding = nn.Parameter(torch.randn(1, 2*out_dim), requires_grad=True)

        self.CA = nn.MultiheadAttention(2*out_dim, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(2*out_dim)
        self.query_projector = Mlp(2*out_dim, 4*out_dim, out_dim, drop=0.1, norm_layer=nn.LayerNorm)
        self.query_scaler_projector = Mlp(2*out_dim, 4*out_dim, out_dim, drop=0.1, norm_layer=nn.LayerNorm)
        self.emb_projector = Mlp(2*out_dim, 4*out_dim, out_dim, drop=0.1, norm_layer=nn.LayerNorm)
        self.emb_scaler_projector = Mlp(2*out_dim, 4*out_dim, out_dim, drop=0.1, norm_layer=nn.LayerNorm)
    
    @autocast()
    def forward(self, **kwargs):
        is_valid_mems = kwargs['lm_emb_valid'].clone()
        first_valid_index = torch.argmax(is_valid_mems.int(), dim=1, keepdim=True)
        # last_valid_index = torch.argmax(valid_mems.flip(1), dim=1, keepdim=True)[0]
        # last_valid_index = valid_mems.shape[1] - last_valid_index - 1
        lm_emb = kwargs['lm_emb']
        mem = torch.stack([lm_emb[i, :, first_valid_index[i]][:,0] for i in range(lm_emb.shape[0])], dim=0)
        mem = self.emb_inprojector(mem)
        # get ego query bias
        tgt = self.pos_embedding.weight
        tgt = tgt.unsqueeze(0).repeat(kwargs['lm_emb'].shape[0], 1, 1)
        # msk = ~kwargs['lm_emb_valid'].bool()
        # mak = torch.stack([msk[i, first_valid_index[i]] for i in range(is_valid_mems.shape[0])], dim=0)
        msk = torch.zeros_like(mem[:,:,0], device=tgt.device).bool()
        for i in range(is_valid_mems.shape[0]):
            msk[i] = ~is_valid_mems[i, first_valid_index[i]].bool().item()

        mem = torch.cat([mem, self.nonsense_embedding.repeat(mem.shape[0], 1, 1)], dim=1)
        msk = torch.cat([msk, torch.zeros(msk.shape[0], 1).bool().to(msk.device)], dim=1)
        output, attn = self.CA(tgt, mem, mem,
                        key_padding_mask=msk,
                        )
        output = self.layer_norm(output)
        # output = self.ego_query_layer_norm(output)
        bev_query_bias = self.query_projector(output)
        bev_query_scalers = self.query_scaler_projector(output)#+1. # TODO
        bev_emb_bias = self.emb_projector(output)
        bev_emb_scalers = self.emb_scaler_projector(output)#+1. # TODO
        return {}, {'bev_query_bias': bev_query_bias,
                    'bev_query_scalers': bev_query_scalers,
                    'bev_emb_bias': bev_emb_bias,
                    'bev_emb_scalers': bev_emb_scalers,
                    }

valid_roller_type = ['map', 'bev', 'bev_wovlm', 'bev_emb', 'bev_emb_wovlm', 'ego_motion', 'motion', 'detr']

@NECKS.register_module()
class RollerBias(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__()
        assert name in valid_roller_type, f"Roller name should be in {valid_roller_type}"
        self.name = name
        in_dim = kwargs['in_dim']
        out_dim = kwargs['out_dim']
        self.in_dim = in_dim
        self.out_dim = out_dim

        roller_num = kwargs['roller_num']
        self.rolling_embedding = nn.Embedding(roller_num, kwargs['in_dim'])
        self.rolling_embedding.weight.requires_grad = True

        self.use_attn_ratio = kwargs.get('use_attn_ratio', False)
        
        self.ca_operator = nn.MultiheadAttention(in_dim, kwargs['nhead'], batch_first=True)
        self.mlp_operator = nn.Sequential(
                                nn.LayerNorm(in_dim),
                                Mlp(in_dim, in_dim*2, in_dim, drop=0.1, act_layer=nn.SiLU),
                                )
        # self.lower_ca_operator = nn.MultiheadAttention(out_dim, kwargs['nhead']//4, batch_first=True)
        # self.lower_mlp_operator = Mlp(out_dim, out_dim*2, out_dim, drop=0.0, norm_layer=nn.LayerNorm)

        self.bias_projector = nn.Linear(in_dim, out_dim)

    def forward(self, mem, msk, **kwargs):

        tgt = self.rolling_embedding.weight.unsqueeze(0).repeat(mem.shape[0], 1, 1)
        o, attn = self.ca_operator(tgt, mem, mem,
                        key_padding_mask=msk,
                        )
        shifted_rolling_embedding = o + self.mlp_operator(o+tgt)   
        bias = None
        bias = self.bias_projector(shifted_rolling_embedding)
        # bias = bias.reshape(mem.shape[0], -1, self.out_dim)
        attn_ratio = 0.0
        if self.use_attn_ratio:
            attn_ratio = attn[:,:,:-1].sum(dim=-1,keepdim=True)/attn.sum(dim=-1,keepdim=True)
        return {
            f'{self.name}_attn_ratio': attn_ratio,
            f'{self.name}_bias': bias,
        }

@NECKS.register_module()
class RollerTF(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__()
        assert name in valid_roller_type, f"Roller name should be in {valid_roller_type}"
        self.name = name
        in_dim = kwargs['in_dim']
        out_dim = kwargs['out_dim']
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.roller_num = kwargs['roller_num']
        self.rolling_embedding = nn.Embedding(self.roller_num, kwargs['in_dim'])
        self.rolling_embedding.weight.requires_grad = True
        
        self.using_nsa = False
        if kwargs.get('attn', None):
            self.using_nsa = True
            self.ca_operator = build_model(kwargs['attn'])
        else:
            self.ca_operator = nn.MultiheadAttention(in_dim, 
                                                    kwargs['nhead'], 
                                                    batch_first=True)
        self.mlp_operator = Mlp(in_dim, in_dim*2, in_dim, drop=0.1, act_layer=nn.SiLU
                                # norm_layer=nn.LayerNorm
                                )
        self.attn_layer_norm = nn.LayerNorm(in_dim)
        nhead = kwargs['nhead'] if out_dim == in_dim else kwargs['nhead']//4
        self.lower_ca_operator = nn.MultiheadAttention(out_dim, nhead, batch_first=True)
        self.lower_mlp_operator = nn.Sequential(
                                nn.LayerNorm(out_dim),
                                Mlp(out_dim, out_dim*2, out_dim, drop=0.1, act_layer=nn.SiLU),
                                    )
        self.mem_projector = nn.Linear(in_dim, out_dim) if not out_dim == in_dim else None

    def forward(self, mem, msk, **kwargs):
        tgt = self.rolling_embedding.weight.unsqueeze(0).repeat(mem.shape[0], 1, 1)
        if self.using_nsa:
            h = self.ca_operator(mem, tgt, mem, mem, return_cache=False)
        else:
            h, attn = self.ca_operator(tgt, mem, mem,
                        key_padding_mask=msk,
                        )
        o = h+self.mlp_operator(self.attn_layer_norm(h))
        shifted_rolling_embedding = o+tgt
        mem_out = self.mem_projector(shifted_rolling_embedding) \
                if self.mem_projector is not None else shifted_rolling_embedding

        return {
            f'{self.name}_mems': mem_out.float(),
            f'{self.name}_ca_operator': self.lower_ca_operator,
            f'{self.name}_mlp_operator': self.lower_mlp_operator,
            'shifted_rolling_embedding': shifted_rolling_embedding
        }

@NECKS.register_module()
class RollerCmp(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__()
        assert name in valid_roller_type, f"Roller name should be in {valid_roller_type}"
        self.name = name
        in_dim = kwargs['in_dim']
        out_dim = kwargs['out_dim']
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nhead = kwargs['nhead']

        self.compress_ratio = 16 # TODO
        self.dim_pe = nn.Embedding(self.compress_ratio, in_dim)
        self.dim_pe.weight.requires_grad=True
        # self.vertical_cmp_mlp = Mlp(self.compress_ratio, 4*self.compress_ratio, 1)
        # self.horizontal_cmp_mlp = Mlp(self.compress_ratio, 4*self.compress_ratio, 1)
        self.vertical_cmp_mlp = nn.ModuleList([
            Mlp(self.compress_ratio, 4*self.compress_ratio, 1) for i in range(self.nhead)
        ])
        self.horizontal_cmp_mlp = nn.ModuleList([
            Mlp(self.compress_ratio, 4*self.compress_ratio, 1) for i in range(self.nhead)
        ])
        
        self.mem_projector = nn.Linear(in_dim, out_dim)

        self.lower_ca_operator = nn.MultiheadAttention(out_dim, kwargs['nhead']//4, batch_first=True)
        self.lower_mlp_operator = nn.Sequential(
                                nn.LayerNorm(out_dim),
                                Mlp(out_dim, out_dim*2, out_dim, drop=0.1, act_layer=nn.SiLU),
                                )


    def forward(self, mem, msk, **kwargs):
        ## roller selected top k
        # is_valid_mems = kwargs['lm_emb_valid'].clone()
        lm_emb_ori = kwargs['lm_emb']
        b, l, _, d = lm_emb_ori.shape
        nonsense_embedding = mem[:,-1:]
        lm_emb = mem[:,:-1].reshape(b, l, -1, d)
        _, _, t, _ = lm_emb.shape

        ### compress along layer
        # v_lm_emb = lm_emb.permute(0,2,1,3).reshape(b, t, -1, self.compress_ratio, d) + self.dim_pe.weight.reshape(1,1,1,self.compress_ratio, d)
        # v_lm_emb = self.vertical_cmp_mlp(v_lm_emb.transpose(-2,-1)).transpose(-2,-1).reshape(b,t,-1,d).permute(0,2,1,3)
        v_lm_emb = lm_emb.permute(0,2,1,3).reshape(b, t, -1, self.compress_ratio, self.nhead, d//self.nhead) \
                    + self.dim_pe.weight.reshape(1,1,1,self.compress_ratio, self.nhead, d//self.nhead)
        v_lm_emb = torch.cat([
            self.vertical_cmp_mlp[i](v_lm_emb[...,i,:].transpose(-2,-1)).transpose(-2,-1)
            for i in range(len(self.vertical_cmp_mlp))
        ], dim=-1).reshape(b,t,-1,d).permute(0,2,1,3)
        
        ### compress along time
        c_l = l//self.compress_ratio
        pad_t = math.ceil(t/self.compress_ratio)*self.compress_ratio

        v_lm_emb = torch.cat([v_lm_emb, 
                        # torch.zeros_like(v_lm_emb,device=v_lm_emb.device)[:,:,:pad_t-t],
                        nonsense_embedding.reshape(b, 1, 1, d).repeat(1, c_l, pad_t-t, 1)
                        ], 
                        dim=-2)
        # h_lm_emb = v_lm_emb.reshape(b, c_l, -1, self.compress_ratio, d)
        # h_lm_emb = self.horizontal_cmp_mlp(h_lm_emb.transpose(-2,-1)).transpose(-2,-1).reshape(b, -1, d)
        h_lm_emb = v_lm_emb.reshape(b, c_l, -1, self.compress_ratio, self.nhead, d//self.nhead)
        h_lm_emb = torch.cat([
            self.horizontal_cmp_mlp[i](h_lm_emb[...,i,:].transpose(-2,-1)).transpose(-2,-1)
            for i in range(len(self.horizontal_cmp_mlp))
        ], dim=-1).reshape(b, -1, d)
        h_lm_emb = self.mem_projector(h_lm_emb)

        return {
            f'{self.name}_mems': h_lm_emb.float(),
            f'{self.name}_ca_operator': self.lower_ca_operator,
            f'{self.name}_mlp_operator': self.lower_mlp_operator,
        }

@NECKS.register_module()
class RollerIter(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__()
        assert name in valid_roller_type, f"Roller name should be in {valid_roller_type}"
        self.name = name
        in_dim = kwargs['in_dim']
        out_dim = kwargs['out_dim']
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nhead = kwargs['nhead']
        self.max_hidden_states_iters = kwargs.get('max_hidden_states_iters', 6)

        self.time_pe = nn.Embedding(self.max_hidden_states_iters, in_dim)
        self.time_pe.weight.requires_grad=False

        self.roller_num = kwargs['roller_num']
        self.rolling_embedding = nn.Embedding(self.roller_num, kwargs['in_dim'])
        self.rolling_embedding.weight.requires_grad = True
        
        self.using_nsa = False
        if kwargs.get('attn', None):
            self.using_nsa = True
            self.ca_operator = build_model(kwargs['attn'])
        else:
            self.ca_operator = nn.MultiheadAttention(in_dim, 
                                                    kwargs['nhead'], 
                                                    batch_first=True)
        self.mlp_operator = Mlp(in_dim, in_dim*2, in_dim, drop=0.1, act_layer=nn.SiLU
                                # norm_layer=nn.LayerNorm
                                )
        self.attn_layer_norm = nn.LayerNorm(in_dim)
        
        self.mem_projector = nn.Linear(in_dim, out_dim)

        self.lower_ca_operator = nn.MultiheadAttention(out_dim, kwargs['nhead']//4, batch_first=True)
        self.lower_mlp_operator = nn.Sequential(
                                nn.LayerNorm(out_dim),
                                Mlp(out_dim, out_dim*2, out_dim, drop=0.1, act_layer=nn.SiLU),
                                )


    def forward(self, mem, msk, **kwargs):
        ## roller selected top k
        # is_valid_mems = kwargs['lm_emb_valid'].clone()
        lm_emb_ori = kwargs['lm_emb']
        b, l, _, d = lm_emb_ori.shape
        last_layer = kwargs.get('last_layer', 0)
        l = last_layer if last_layer else l
        nonsense_embedding = mem[:,-1:]
        lm_emb = mem[:,:-1].reshape(b, l, -1, d)
        _, _, t, _ = lm_emb.shape

        lm_emb = torch.cat([lm_emb, nonsense_embedding.repeat(1,1,t,1)], dim=1)

        lm_emb = lm_emb.permute(0,2,1,3)
        lm_emb = lm_emb + self.time_pe.weight[range(lm_emb.shape[1])][None, :, None, :]
        lm_emb = lm_emb.flatten(0,1)
        q = self.rolling_embedding.weight.unsqueeze(0).repeat(lm_emb.shape[0],1,1)
        h,_ = self.ca_operator(q, lm_emb, lm_emb)
        h = h + q
        h = h + self.mlp_operator(self.attn_layer_norm(h))
        h = h.reshape(b, t, self.roller_num, d)
        h = h.mean(dim=1)
        h = self.mem_projector(h)
        return {
            f'{self.name}_mems': h.float(),
            f'{self.name}_ca_operator': self.lower_ca_operator,
            f'{self.name}_mlp_operator': self.lower_mlp_operator,
        }
        

@NECKS.register_module()
class RollingAdaptor(RollNetAdaptorBase):
    def __init__(self, **kwargs):
        super().__init__()
        h, w = kwargs['bev_h'], kwargs['bev_w']
        in_dim = kwargs['in_dim']
        # out_dim = kwargs['out_dim']
        # assert in_dim%out_dim == 0, f'in_dim {in_dim} should be divisible by out_dim {out_dim}'
        self.in_dim = in_dim
        # self.out_dim = out_dim
        self._h, self._w = h, w

        self.all_iter = kwargs.get('all_iter', False)
        self.last_layer = kwargs.get('last_layer', 1)

        ### a roller 
        rollers = []
        # for name in ['map', 'bev', 'bev_emb', 'motion', 'detr']:
        for name in kwargs['rolling_query_names']:
            roller_cfg = kwargs['roller']
            roller_cfg.update({
                'name': name,
                'max_hidden_states_iters': kwargs.get('max_hidden_states_iters', 6),
            })
            rollers.append(build_model(roller_cfg))

        self.rollers = nn.ModuleList(rollers)

        self.nonsense_embedding = nn.Parameter(torch.randn(1, in_dim), requires_grad=True)

        self.use_layer_embedding = kwargs.get('use_layer_embedding', True)
        if self.use_layer_embedding:
            self.layer_embedding = nn.Embedding(kwargs['tf_layer_num'], in_dim)
            self.layer_embedding.weight.requires_grad=True

        self.max_hidden_states_iters = kwargs.get('max_hidden_states_iters', 6)
        

    @autocast()
    def forward(self, **kwargs):
        is_valid_mems = kwargs['lm_emb_valid'].clone()
        first_valid_index = torch.argmax(is_valid_mems.int(), dim=1, keepdim=True)
        last_valid_index = torch.argmax(is_valid_mems.int().flip(1), dim=1, keepdim=True)
        last_valid_index = is_valid_mems.shape[1] - last_valid_index
        lm_emb = kwargs['lm_emb']
        if self.use_layer_embedding:
            lm_emb = lm_emb + self.layer_embedding.weight[None,:,None,:]
        # extract mems
        out = {}
        mem_list = []
        msk_list = []
        layer = -self.last_layer if self.last_layer else None
        assert self.last_layer >= 0, f'num last_layer {self.last_layer} should be >= 0'
        valid_iters = last_valid_index.max() - first_valid_index.min()
        valid_iters = min(self.max_hidden_states_iters, valid_iters)
        for offset in range(valid_iters):
            mem = torch.stack([lm_emb[i, layer:, first_valid_index[i,0]+offset]
                                for i in range(lm_emb.shape[0])], dim=0)
            msk = torch.zeros_like(mem[:,:,0], device=mem.device).bool()
            for i in range(mem.shape[0]):
                msk[i] = ~is_valid_mems[i, first_valid_index[i,0]+offset].bool()

            mem_list.append(mem)
            msk_list.append(msk)

            if offset == valid_iters-1:
                mem = torch.cat(mem_list, dim=1)
                msk = torch.cat(msk_list, dim=1)
                mem = torch.cat([mem, self.nonsense_embedding.repeat(mem.shape[0], 1, 1)], dim=1)
                msk = torch.cat([msk, torch.zeros(msk.shape[0], 1, device=mem.device).bool()], dim=1)
                for r in self.rollers:
                    out.update(r(mem, msk, last_layer=self.last_layer, **kwargs))
                    

        # if not self.all_iter:
        #     mem = torch.stack([lm_emb[i, layer:, first_valid_index[i]][:,0] 
        #                         for i in range(lm_emb.shape[0])], dim=0)
        #     msk = torch.zeros_like(mem[:,:,0], device=mem.device).bool()
        #     for i in range(is_valid_mems.shape[0]):
        #         msk[i] = ~is_valid_mems[i, first_valid_index[i]].bool().item()

        #     mem = torch.cat([mem, self.nonsense_embedding.repeat(mem.shape[0], 1, 1)], dim=1)
        #     msk = torch.cat([msk, torch.zeros(msk.shape[0], 1, device=mem.device).bool()], dim=1)
        #     for r in self.rollers:
        #         out.update(r(mem, msk))
        return {}, out
