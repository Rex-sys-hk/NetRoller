import time
import copy
import torch
from torch import nn
from torch.cuda.amp import autocast
import numpy as np
import pickle

from mmdet.models import DETECTORS
from mmdet.models.detectors import BaseDetector
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
# from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
import time

@DETECTORS.register_module()
class RollNet(BaseDetector):
    def __init__(self, 
            level_of_models:list,
            level_of_adaptors:list,
            **kwargs,
            ):
        super(RollNet, self).__init__()
        # self.level_of_models = level_of_models
        # print(level_of_models)
        self.level_of_models = nn.ModuleList([
            build_model(m_cfg, 
            # train_cfg=m_cfg.get('train_cfg'), test_cfg=m_cfg.get('test_cfg')
            ) 
            for m_cfg in level_of_models
        ])
        self.level_of_adaptors = nn.ModuleList([
            build_model(am_cfg) for am_cfg in level_of_adaptors   
        ])
        self.pretrained = kwargs['pretrained']
        self.pretrained_adaptor = kwargs['pretrained_adaptor']
        self.max_gen_len = kwargs.get('max_gen_len', 256)
        self.d_time_list = {}
        for i in range(len(self.level_of_models)):
            self.d_time_list[f'dt_level_{i}'] = []
    
    # @autocast()
    def forward(self, *args, **kwargs):
        # print(kwargs.keys())
        # x = kwargs['x']
        kwargs.update({'max_gen_len':self.max_gen_len})
        loss = {}
        outputs = {}
        for i in range(len(self.level_of_models)):
            if not self.training:
                torch.cuda.synchronize()
                t_start = time.time()
            l,o = self.level_of_models[i](
                **outputs, 
                **kwargs)
            loss.update(l)
            outputs.update(o)
            l,o = self.level_of_adaptors[i](
                **outputs, 
                **kwargs)
            loss.update(l)
            outputs.update(o)
            if not self.training:
                torch.cuda.synchronize()
                self.d_time_list[f'dt_level_{i}'].append(time.time() - t_start)
        if not self.training:
            self.show_time()
        if self.training:
            return loss
        if kwargs.get('return_all', False):
            return outputs
        outputs['bbox_results'][0]['pts_bbox']['prompts'] = kwargs['prompts']
        outputs['bbox_results'][0]['pts_bbox']['decoded'] = outputs['decoded']
        print('')
        return outputs['bbox_results']
    
    def init_weights(self):
        # pass
        for m, c in zip(self.level_of_models, self.pretrained):
            if c is not None:
                # load_checkpoint(m, c, map_location='cpu')
                m.load_model(c, map_location='cpu')
        for m, c in zip(self.level_of_adaptors, self.pretrained_adaptor):
            if c is not None:
                # load_checkpoint(m, c, map_location='cpu')
                m.load_model(c, map_location='cpu')
    
    def set_epoch(self, epoch): 
        for m in self.level_of_models:
            try:
                m.pts_bbox_head.epoch = epoch
            except:
                pass

    def extract_feat(self, imgs):
        """Extract features from images."""
        # pass
        raise NotImplementedError


    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError

    def show_time(self, save_path=None):
        for i in range(len(self.level_of_models)):
            dts = np.array(self.d_time_list[f'dt_level_{i}'])
            print(f'dt_level_{i}: mean: {dts.mean()}, std: {dts.std()}, min: {dts.min()}, max: {dts.max()}')
        if save_path is not None:
            with open(save_path+'/latency.pkl', 'a') as f:
                pickle.dump(self.d_time_list, f)