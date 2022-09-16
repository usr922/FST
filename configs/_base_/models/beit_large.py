_base_ = [
    '../../_base_/models/upernet_beit.py'
]

model = dict(
    pretrained='pretrained/beit_large_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        type='BEiT',
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        qv_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23]),
    neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024], num_classes=19, channels=1024),
    auxiliary_head=dict(in_channels=1024, num_classes=19),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))
