import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import projects.mmdet3d_plugin.llama.utils as llama_utils
from projects.mmdet3d_plugin.llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
import cv2
import re

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# create data
transform_train = transforms.Compose([
    transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ), # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class FinetuneDataset:
    def __init__(self, transform, max_words=512, tokenizer_path=None, 
                text_desc_types = ['perception', 'prediction', 'planning', 'behavior'],
                text_latest_frame = False):
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        self.latest_frame = text_latest_frame
        self.desc_types = text_desc_types

    def get_text_label(self, scene_labels, cur_token):
        outputs = {}
        latency_offset = 3 # 1,2,3
        # fame with contents
        key_frames = scene_labels['frames']
        # frame token list
        frame_token_list = scene_labels['frame_token_list']
        # get valid frame token list before cur_token
        cur_frame_idx = frame_token_list.index(cur_token)-latency_offset
        key_frame_idx = scene_labels['key_frame_idx']
        valid_frame_idx = [k for k in key_frame_idx if k <= cur_frame_idx]
        valid_frame_tokens = [frame_token_list[k] for k in valid_frame_idx]

        # if 1:
        if len(valid_frame_tokens) == 0:
            outputs['has_text_label'] = False
            outputs['timestamp'] = 0
            outputs['text_token'] = torch.zeros(self.max_words, dtype=torch.int64)
            outputs['text_labels'] = torch.zeros(self.max_words, dtype=torch.int64)
            outputs['text_imgs'] = torch.zeros(6, 3, 224, 224)
            # outputs['text_token_mask'] = torch.zeros(self.max_words, dtype=torch.float32)
            outputs['qa_type'] = torch.tensor([-1])
            outputs['prompts'] = torch.zeros(self.max_words//2, dtype=torch.int64)
            outputs['prompts_mask'] = torch.zeros(self.max_words//2, dtype=torch.float32)
            return outputs

        k_frame_token = valid_frame_tokens[-1] if self.latest_frame else random.sample(valid_frame_tokens,1)[0]
        k_frame = key_frames[k_frame_token]

        # ['perception', 'prediction', 'planning', 'behavior']
        # self.desc_types=['perception']
        # self.desc_types=['prediction']
        # self.desc_types=['planning']
        # self.desc_types=['behavior']
        desc_type = random.sample(self.desc_types,1)[0]
        desc_type_id = self.desc_types.index(desc_type)

        data_item = random.sample(k_frame[desc_type],1)[0]
        # data_item = k_frame[desc_type][0]
        
        outputs['timestamp'] = k_frame['timestamp']
        if 'image' in data_item.keys():
            filename = data_item['image']
            question = data_item['conversations'][0]['value']
            answer = data_item['conversations'][1]['value']
            if isinstance(filename, list):
                image_all = []
                for img_path in filename:
                    image = cv2.imread(img_path)
                    image = Image.fromarray(image)
                    image = self.transform(image)
                    image_all.append(image)
                image = torch.stack(image_all)
            else:
                image = cv2.imread(filename)
                image = Image.fromarray(image)
                image = self.transform(image)
            format_instruction = question
            format_input = None
        else:
            image = torch.zeros(3, 224, 224)
            format_instruction = data_item['instruction']
            format_input = data_item['input']
            answer = data_item['output']
        input1 = llama_utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        # print('inpu1:', input1)
        # print('input2:', input2)
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        prompts = input1.clone()
        # padding = self.max_words//2 - input1.shape[0]
        # if padding > 0:
        #     prompts = torch.cat((torch.zeros(padding, dtype=torch.int64) - 1, prompts))
        # elif padding < 0:
        #     prompts = prompts[-self.max_words//2:]
        prompts_mask = prompts.ge(0)

        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        # input2_mask = input2_mask.float()
        # label_mask = label_mask.float()

        outputs['has_text_label'] = True
        outputs['text_token'] = input2
        outputs['text_labels'] = labels
        outputs['text_imgs'] = image
        # outputs['text_token_mask'] = input2_mask
        outputs['qa_type'] = torch.tensor([desc_type_id])
        outputs['prompts'] = prompts
        outputs['prompts_mask'] = prompts_mask.float()

        return outputs
        # return input2, labels, input2_mask, image

class PretrainDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        images, captions = [], []
        for meta_path in self.config['META']:
            images_this_meta, captions_this_meta = [], []
            for chunk in pd.read_csv(meta_path, sep='\t', lineterminator='\n', chunksize=10 ** 6):
                images_this_meta.extend(chunk['url'].tolist())
                captions_this_meta.extend(chunk['caption'].tolist())
            print(f"{meta_path}: len {len(images_this_meta)}")
            images.extend(images_this_meta)
            captions.extend(captions_this_meta)

        self.data_list = []
        for x, y in zip(images, captions):
            self.data_list.append({'url': x, 'caption': y})
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, caption = sample['url'], sample['caption']
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        image = cv2.imread(image_path)
        image = Image.fromarray(image)
        image = self.transform(image)

        format_instruction = "Generate caption of this image"
        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image
