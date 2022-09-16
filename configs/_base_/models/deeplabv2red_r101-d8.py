_base_ = ['deeplabv2_r50-d8.py']
# Previous UDA methods only use the dilation rates 6 and 12 for DeepLabV2.
# This might be a bit hidden as it is caused by a return statement WITHIN
# a loop over the dilation rates:
# https://github.com/wasidennis/AdaptSegNet/blob/fca9ff0f09dab45d44bf6d26091377ac66607028/model/deeplab.py#L116
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
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
    decode_head=dict(dilations=(6, 12)))