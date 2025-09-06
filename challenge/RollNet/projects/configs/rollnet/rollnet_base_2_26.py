_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
# The number of GPUs used for training
GPU_num = 8

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

# img_norm_cfg=dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg=dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_classes = len(class_names)

# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.
## TODO: Study params
total_epochs = 5
cumulative_iters = 8
used_data_ratio = 1.0

VAD_phase = 'freeze'
# llama_phase = 'finetune'
llama_phase = 'freeze'

fut_mode = 6 if VAD_phase=='freeze' else 6

text_desc_types = ['perception', 'prediction', 'planning', 'behavior']
# text_desc_types = ['perception', 'prediction']
# text_desc_types = ['planning', 'behavior']
text_latest_frame = True
lr_scale = 1.0
use_ds = False # weather use deepspeed. Not work
roller_num = 1
roller_out_dim = 1*_dim_
rolling_query_names = ['bev'] # ['map', 'bev', 'bev_emb', 'motion', 'ego_motion', 'detr']
roller_type = "RollerBias" # RollerIter, RollerTF, RollerBias
use_layer_embedding = False
use_attn_ratio = False
tf_layer_num = 32
all_iter = True # unused, deprecated
max_hidden_states_iters = 192 # 192
max_gen_len = max_hidden_states_iters
last_layer = 0


# import sys 
# sys.path.append('./projects/configs')
# from VAD.VAD_base_stage_2 import model as vad_model
# from lm.llama_fintune import model as lm_model
load_lm_from='/mnt/csi-data-aly/shared/public/xinren/DriveLM/challenge/llama_adapter_v2_multimodal7b/output/checkpoint-3.pth'
load_ad_from='/mnt/csi-data-aly/shared/public/xinren/DriveLM/challenge/RollNet/ckpt/VAD_base.pth'

llama_path = '/mnt/csi-data-aly/shared/public/xinren/Llama-2-7b'
lm_model = dict(
    type='LLaMA_adapter',
        llama_ckpt_dir=llama_path+'/7B',
        llama_tokenizer=llama_path+'/tokenizer.model',
        max_seq_len=512, 
        max_batch_size=32,
        # clip_model='ViT-L/14',
        clip_model='/mnt/csi-data-aly/shared/public/xinren/clip/ViT-L-14.pt',
        v_embed_dim=768, 
        v_depth=8,
        v_num_heads=16, 
        v_mlp_ratio=4.0,
        query_len=10, 
        query_layer=31,
        w_bias=False, 
        w_lora=False, 
        lora_rank=16, 
        w_new_gate=False,
        # phase="freeze", # TODO
        phase=llama_phase,
        # quat='fp32',
)
ad_model = dict(
    type='VAD',
    use_grid_mask=True,
    video_test_mode=True,
    phase=VAD_phase, # TODO
    in_rollnet=True,
    fut_mode=fut_mode, # TODO
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='VADHead',
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        tot_epoch=total_epochs,
        use_traj_lr_warmup=False,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_his_encoder=None,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        fut_mode=fut_mode, # TODO
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        use_pe=True,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=300,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        map_num_vec=map_num_vec,
        map_num_classes=map_num_classes,
        map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
        map_num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='VADPerceptionTransformer',
            map_num_vec=map_num_vec,
            map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            map_decoder=dict(
                type='MapDetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=num_classes),
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=map_num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_traj=dict(type='L1Loss', loss_weight=0.2),
        loss_traj_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.2),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0),
        loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=1.0, dis_thresh=1.0),
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.5)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),
        map_assigner=dict(
            type='MapHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
            pc_range=point_cloud_range))))

adaptor_model_end=dict(
    type='RollNetAdaptorBase',
)

adaptor_model_0_1=dict(
    type='RollingAdaptor',
    bev_h=bev_h_,
    bev_w=bev_w_,
    in_dim=4096,
    rolling_query_names=rolling_query_names, # ['map', 'bev', 'bev_emb', 'motion', 'detr']
    roller = dict(
        type=roller_type,
        name=None,
        in_dim=4096,
        out_dim=roller_out_dim,
        nhead=32,
        roller_num=roller_num, # must be 1 if use SparseAttention
        use_attn_ratio=use_attn_ratio,
        # attn=dict(
        #         type='SparseAttention',
        #         dim=4096, 
        #         dim_head=128, 
        #         heads=32,
        #         sliding_window_size=8,
        #         compress_block_size=8,
        #         compress_block_sliding_stride=4,
        #         selection_block_size=4,
        #         num_selected_blocks=8,
        #         norm=False,
        #         causal=False,
        #         )
        ),
    max_hidden_states_iters=max_hidden_states_iters,
    use_layer_embedding=use_layer_embedding,
    tf_layer_num=tf_layer_num,
    all_iter=all_iter,
    last_layer=last_layer, # last num of layers, 0 means all
)

model = dict(
    type='RollNet',
    level_of_models=[lm_model, ad_model],
    level_of_adaptors=[adaptor_model_0_1, adaptor_model_end],
    pretrained=[load_lm_from, load_ad_from],

    # level_of_models=[ad_model],
    # level_of_adaptors=[adaptor_model_end],
    # pretrained=[load_ad_from],

    # level_of_models=[lm_model],
    # level_of_adaptors=[adaptor_model_end],
    # pretrained=[load_lm_from],

    pretrained_adaptor=[None, None],
    max_gen_len=max_gen_len,
    train_cfg=[],
    test_cfg=[],
    )

# dataset_type = 'VADCustomNuScenesDataset'
dataset_type = 'RollNetCustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D',\
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=5,
         use_dim=5,
         file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_label=False, with_ego=True),
            dict(type='CustomCollect3D',\
                 keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                       'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                       'ego_lcf_feat', 'gt_attr_labels'])])
]

data = dict(
    samples_per_gpu=1, # can only be 1 for both training and testing
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'rollnet_nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        map_classes=map_classes,
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        custom_eval_version='vad_nusc_detection_cvpr_2019',
        max_words=192,
        tokenizer_path=llama_path+'/tokenizer.model',
        used_data_ratio=used_data_ratio,
        text_desc_types=text_desc_types,
        text_latest_frame=text_latest_frame,
        ),
    val=dict(type=dataset_type,
             data_root=data_root,
             pc_range=point_cloud_range,
             ann_file=data_root + 'rollnet_nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,
             map_classes=map_classes,
             map_ann_file=data_root + 'nuscenes_map_anns_val.json',
             map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
             map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
             use_pkl_result=True,
             custom_eval_version='vad_nusc_detection_cvpr_2019',
             max_words=192,
             tokenizer_path=llama_path+'/tokenizer.model',
             used_data_ratio=used_data_ratio,
             text_desc_types=text_desc_types,
             text_latest_frame=text_latest_frame,
             ),
    test=dict(type=dataset_type,
              data_root=data_root,
              pc_range=point_cloud_range,
              ann_file=data_root + 'rollnet_nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality, samples_per_gpu=1,
              map_classes=map_classes,
              map_ann_file=data_root + 'nuscenes_map_anns_val.json',
              map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
              map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
              use_pkl_result=True,
              custom_eval_version='vad_nusc_detection_cvpr_2019',
              max_words=192,
              tokenizer_path=llama_path+'/tokenizer.model',
              used_data_ratio=used_data_ratio,
              text_desc_types=text_desc_types,
              text_latest_frame=text_latest_frame,     
              ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
    type='AdamW',
    lr=lr_scale*1e-3*data['samples_per_gpu']*GPU_num/256,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    betas=(0.9, 0.95),
    weight_decay=0.02,
    )

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    grad_clip=dict(max_norm=5, norm_type=2),
    cumulative_iters=cumulative_iters,
    )
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    # warmup_iters=int(27976/(data['samples_per_gpu']*2)),
    warmup_by_epoch=True,
    warmup_iters=1,
    # warmup_ratio=1.0 / 3,
    warmup_ratio=1e-3,
    min_lr_ratio=1e-6,
    )

evaluation = dict(interval=total_epochs, pipeline=test_pipeline, metric='bbox', map_metric='chamfer')

#### use basic runner
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# #### use deepspeed
strategy=dict(
    type='DeepSpeedStrategy',
    fp16=dict(
        enabled=True,
        auto_cast=True,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=1e3,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=24,
    ),
    # amp=dict(enabled=True, opt_level='01'),
    inputs_to_half=[0],
    zero_optimization=dict(
        stage=2,
        allgather_partitions=True,
        reduce_scatter=True,
        allgather_bucket_size=5e6,
        reduce_bucket_size=5e6,
        overlap_comm=True,
        contiguous_gradients=True,
        offload_optimizer={
            "device": "cpu",
            "pin_memory": True,
        },
        offload_param={
            "device": "cpu",
            "pin_memory": True,
        },
        # zero_quantized_weights=True,
        # stage3_param_persistence_threshold=1e7,
        ),
    train_micro_batch_size_per_gpu=1,
    gradient_accumulation_steps=1,
    steps_per_print=10, 
    gradient_clipping=1.0,
    scheduler={
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 5e-6,
            "warmup_max_lr": 1e-3*data['samples_per_gpu']*GPU_num/256,
            "warmup_num_steps": 3497,
            "total_num_steps": 3497*total_epochs,
        },
    },
    optimizer={
        "type": "Adam",
        "params": {
            "lr": 1e-3*data['samples_per_gpu']*GPU_num/256,
            "betas": [
                0.9,
                0.95
            ],
            "eps": 1e-5,
            "weight_decay": 0.01,
            "torch_adam":True,
            "adam_w_mode":True,
        },
    },
)
# load_from = 'ckpts/VAD_base_stage_1.pth'
# load_from = [load_lm_from, load_ad_from]
# load_from = 'outputs/250401151239/latest.pth'
# resume_from = "./outputs/250407154304/latest.pth"
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# fp16 = dict(loss_scale=512.)
# find_unused_parameters = True
checkpoint_config = dict(interval=1, max_keep_ckpts=3) # total_epochs


custom_hooks = [dict(type='CustomSetEpochInfoHook')]