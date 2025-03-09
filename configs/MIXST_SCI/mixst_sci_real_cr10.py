_base_=[
        "../_base_/real_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]

cr = 10
resize_h,resize_w = 128,128

train_pipeline = [ 
    dict(type='RandomResize'),
    dict(type='RandomCrop',crop_h=resize_h,crop_w=resize_w,random_size=True),
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Flip', direction='diagonal',flip_ratio=0.5,),
    dict(type='Resize', resize_h=resize_h,resize_w=resize_w),
]

gene_meas = dict(type='GenerationGrayMeas')

train_data = dict(
    type="DavisData",
    data_root="/home/wanglishun/datasets/DAVIS/DAVIS-480/JPEGImages/480p",
    mask_path="test_datasets/mask/real_mask.mat",#
    mask_shape=(resize_h,resize_w,cr),
    pipeline=train_pipeline,
    gene_meas = gene_meas,
)

real_data = dict(
    data_root="test_datasets/real_data/cr10",
    cr=cr
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

opt = dict(
    spatial_branch=True,
    spectral_branch=True,
    spatial_interaction=True,
    spectral_interaction=True,
    stage_interaction=True,
    block_interaction=True,
    bias=False,
    in_dim=2, 
    out_dim=2,
    dim=32, #32, #64
    stage=5,
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
    share_params=True
)

eval=dict(
    flag=False,
    interval=1
)

checkpoints=None#"checkpoints/stformer/stformer_base_real_cr10.pth"