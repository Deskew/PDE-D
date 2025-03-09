_base_=[
        "../_base_/six_gray_sim_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
)

resize_h,resize_w = 192,192#256,256#128,128#
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
test_data = dict(
    mask_path="test_datasets/mask/new_mask.mat"#random_mask.mat"#matrix10_1.mat"#matrix_1.mat" #
)

opt = dict(
    spatial_branch=True,
    temporal_branch=True,
    temporal_interaction=True,
    spatial_interaction = False,
    stage_interaction=True,
    block_interaction=True,
    bias=False,
    in_dim=1, 
    out_dim=1,
    cr=8,
    dim=32,#64,#48, #16, #
    stage=5,#9,#
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
    color_channels=1, 
    share_params=False#True#
)

eval=dict(
    flag=True,
    interval=1
)

checkpoints="checkpoints/pde-d/PDE-D-9st_gray.pth" #work_dirs/mixst_sci/checkpoints/epoch_7.pth"#"/home/yutang/Documents/sci/DADUN/None#