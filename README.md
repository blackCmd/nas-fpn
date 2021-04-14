Detection and tracking for swooned person using NAS-FPN (Ver. 2.0)
========

Introduce
-----
This repo is the output of the [IITP AI Grand Challenge](http://www.ai-challenge.kr/). It was released under [Apache License 2.0](https://github.com/blackCmd/nas-fpn/blob/master/LICENSE), and anyone can use it freely within the scope of the license.

![demo image](resources/sample_slow.gif)

How to use
-------
1. Download this repo and build. (Build tutorial is [here](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).)
2. Make *"**./checkpoints**"* directory and download model below  ***./checkpoints**.*
3. run ```python ./demo.py```.
4. You can see result like below.
![result_image](resources/inference_result.jpeg)


Model Zoo
-------
https://drive.google.com/file/d/1pbc1R4oIaKNTQ-R_kqx9QnBmziN9up6B/view?usp=sharing

(Download below **./checkpoints/**)

과제를 위해 수행한 연구들
-------
1. 데이터셋 수집 및 라벨링   
 AI HUB 실신 이상행동/동작 동영상 데이터셋 약 600GB 수집.   
 4차 2단계 대회의 샘플 실신 이미지 데이터셋 약 24,400개 수집.   
 구글링으로 찾은 실신 데이터셋 약 600개 수집.   
 기타 대학원, 연구소 등의 데이터셋 60,000개 수집(UMA fall dataset, SLP fall dataset, Human posture dataset 등등).   
 yolo/coco type으로 모든 데이터셋 필터링 및 라벨링.

2. Data Augmentation   
 Mosaic, Mixup, Flip 등의 Augumentation 기법을 사용하여 학습데이터 증강

3. Hyper-Parameter와 Resolution을 변경 후 학습시도(아래 config 코드는 ./config/nas-fpn 아래의 .py 파일내에서 사용가능)   
 ```
 _base_ = [
    '../_base_/models/my_retinanet_r50_fpn.py',
    '../_base_/datasets/my_coco_detection.py', '../_base_/default_runtime.py'
]
cudnn_benchmark = True
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    neck=dict(type='NASFPN', stack_times=7, norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg))
# training and testing settings
train_cfg = dict(assigner=dict(neg_iou_thr=0.5))
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),    
    dict(
        type='Resize',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(640, 640)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    # lr=0.08,#원본
    lr=0.00325,    #파인 튜닝용
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,  #원본
    warmup_ratio=0.1,   #원본     
    # step=[30, 40]) #원본
    step=[7]) #원본
# runtime settings
#total_epochs = 50 #원본
total_epochs = 8

 ```
 
 4. 학습 결과   
 ![mAP](/resources/mAP.png)
 ![FPS](/resources/FPS.png)

Reference
-----
[NAS-FPN](https://arxiv.org/abs/1904.07392)

[MMDetectoin](https://github.com/open-mmlab/mmdetection)


