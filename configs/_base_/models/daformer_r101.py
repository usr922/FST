_base_ = ['../../_base_/models/daformer_sepaspp_mitb5.py']

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    # pretrained='open-mmlab://resnet101_v1c',
    pretrained='pretrained/resnet101_v1c-e67eebb6.pth',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    # model training and testing settings
    decode_head=dict(
        in_channels=[256, 512, 1024, 2048],
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
