_base_=[
        "../_base_/variable_cr_bayerdata.py",
        "../_base_/default_runtime.py"
        ]
resize_h,resize_w = 512,512#,256,256#,
cr = 8#10 
gene_meas = dict(type='GenerationBayerMeas')

test_data = dict(
    mask_shape = (resize_h,resize_w,cr),
    data_root="/home/yutang/Documents/sci/DADUN/test_datasets/variable_cr_data/Parkour/color_512",
    mask_path="test_datasets/mask/mid_color_mask.mat",# new_color_mask.mat",#  #mask.mat"
    gene_meas = gene_meas,
)

data = dict(
    samples_per_gpu=1, #1,5
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
    color_channels=3, 
    share_params=False#True#
)

eval=dict(
    flag=True,
    interval=1
)

checkpoints=None#