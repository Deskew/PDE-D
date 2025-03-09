_base_=[
        "../_base_/variable_cr_data.py",
        "../_base_/default_runtime.py"
        ]
resize_h,resize_w = 512,512#,256,256#,
cr = 8#10 
gene_meas = dict(type='GenerationGrayMeas')

test_data = dict(
    mask_shape = (resize_h,resize_w,cr),
    data_root="/home/yutang/Documents/sci/DADUN/test_datasets/variable_cr_data/Parkour/gray_512",#gray_256",#
    mask_path="test_datasets/mask/gray_512_mask.mat",#new_mask.mat",#random_mask.mat", #
    gene_meas = gene_meas,
)

data = dict(
    samples_per_gpu=1, #1,5
    workers_per_gpu=1,
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
    stage=9,#5,#
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

checkpoints=None#