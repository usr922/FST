_base_ = [
    '../../_base_/models/upernet_beit.py'
]

model = dict(
    pretrained='pretrained/beit_base_patch16_224_pt22k_ft22k.pth',
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426))
    )

