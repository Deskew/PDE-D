_base_=[
        "../_base_/davis_bayer.py",
        "../_base_/matlab_bayer.py",
        "../_base_/default_runtime.py"
        ]
test_data = dict(
    data_root="test_datasets/middle_scale",
    mask_path="test_datasets/mask/new_color_mask.mat",#mid_color_mask.mat",#
    rot_flip_flag=True
)
resize_h,resize_w = 192,192
train_pipeline = [ 
    dict(type='RandomResize'),
    dict(type='RandomCrop',crop_h=resize_h,crop_w=resize_w,random_size=True),
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Flip', direction='diagonal',flip_ratio=0.5,),
    dict(type='Resize', resize_h=resize_h,resize_w=resize_w),
]
train_data = dict(
    mask_path = None,
    mask_shape = (resize_h,resize_w,8),
    pipeline = train_pipeline
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)

opt = dict(
    spatial_branch=True,
    temporal_branch=True,
    temporal_interaction=True,
    spatial_interaction = False,
    stage_interaction=True,
    block_interaction=True,
    bias=False,
    in_dim=3, 
    out_dim=3,
    cr=8,
    dim=32,#48, #16, #32,#
    stage=5,#5,#
    DW_Expand=1, #expand of depth-wise convolution
    FFN_Expand=2.66,
    share_params=1, # whether stage share parameters
    ffn_name='Gated_Dconv_FeedForward',
    LayerNorm_type='BiasFree',
    act_fn_name='gelu'   
)
model = dict(
    type='MixST_SCI',
    opt=opt,
    color_channels=3, 
    share_params=False#True#
)
 
eval=dict(
    flag=False,#True, #
    interval=1
)
checkpoints="/home/yutang/Documents/sci/DADUN/work_dirs/mixst_sci_color/ch32-5stages-color/epoch_126.pth"#None#