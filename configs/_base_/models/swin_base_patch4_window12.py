_base_ = [
    '../../_base_/models/upernet_swin.py',
]

# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa

# checkpoint_file = 'pretrained/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'
checkpoint_file = 'pretrained/swin_base_patch4_window12_384.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=19),
    auxiliary_head=dict(in_channels=512, num_classes=19))
