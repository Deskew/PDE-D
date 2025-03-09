import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from cacti.datasets.builder import build_dataset 
from cacti.models.builder import build_model
from cacti.utils.optim_builder import  build_optimizer
from cacti.utils.loss_builder import build_loss
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks, generate_cr_masks
from cacti.utils.config import Config
from cacti.utils.logger import Logger
from cacti.utils.utils import save_image, load_checkpoints, get_device_info
from cacti.utils.test_variable import eval_psnr_ssim
from cacti.utils.learnig_rate_schedulers import WarmupStepLR

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.cuda import amp


import time
import argparse 
import json 
import einops

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("config",type=str)
    #gray:
    #5st-32ch:256:32.82/0.920; 512:34.08/0.945;  
    #9st-32ch:256:33.22/0.926; 512:34.45/0.949;  
    #color:
    #5st-32ch:512: 34.13/0.945;
    parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/MIXST_SCI/mixst_sci_variable_cr.py')
    #5-stages-gray
    #parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/pde-d/PDE-D-5st_gray.pth')
    #9-stages-gray
    parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/pde-d/PDE-D-9st_gray.pth')
    #10-iter-256:32.83/0.919 512:34.13/0.945
    #5-iter-256:32.49/0.914 512:33.82/0.942
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/DUN_RNNViT/brdun_rnnvit_variable_cr.py')
    # parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/dadun_5i64ch_gray.pth')
    
    #256:32.41/0.914; 512:33.79/0.941;  
    #parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/STFormer/stformer_variable_cr.py') 
    #parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/stformer/stformer_base.pth')

    #256:32.24/0.909; 512:33.46/0.936;  
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/ELP-Unfolding/elpunfolding_variable_cr.py') 
    # parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/elpunfolding/elpunfolding.pth')

    #256:32.70/0.919; 512:25.54/0.779;  
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/DUN-3DUnet/dun3dunet_variable_cr.py') 
    # parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/dun3dunet/dun3dunet.pth')

    #256:31.26/0.896; 512:14.32/0.249;  
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/RevSCI/revsci_variable_cr.py') 
    # parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/revsci/revsci.pth')

    #256:30.65/0.883; 512:16.89/0.343;  
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/BIRNAT/birant_variable_cr.py') 
    # parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/birnat/birnat.pth')

    #256:28.82/0.836; 512:15.09/0.219;  
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/GAP-net/gapnet_variable_cr.py') 
    # parser.add_argument("--weights",type=str, default='checkpoints/gapnet/gapnet.pth')
    
    #256:32.73/0.918; 512:34.07/0.945;   
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/EfficientSCI/efficientsci_variable_cr.py')
    # parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/efficientsci/efficientsci_base.pth')

    #256:32.80/0.919; 512:34.10/0.945;  
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/EfficientSCI_plus_plus/efficientsci_plus_plus_variable_cr.py') 
    # parser.add_argument("--weights",type=str, default='checkpoints/efficientsci_plus_plus/efficientsci_plus_plus_base.pth')

#######################################Color########################################
    
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/MIXST_SCI/mixst_sci_variable_cr_color.py')
    # #5-stages-color:
    # #5st-32ch:512: 31.61/0.915; #new mask:30.01/0.896
    # parser.add_argument("--weights",type=str, default='work_dirs/mixst_sci_color/checkpoints/epoch_1.pth')
    #parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/pde-d/PDE-D-5st_mid_color.pth')
    #9-stages-color 512:
    #parser.add_argument("--weights",type=str, default='/home/yutang/Documen ts/sci/DADUN/checkpoints/mixst_sci/ch32-9stagesepoch_8.pth')
   
    
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/BRDUN_RNNViT/brdun_rnnvit_variable_cr_color.py')
    # parser.add_argument("--weights",type=str, defa ult='/home/yutang/Documents/sci/DADUN/work_dirs/brdun_rnnvit/checkpoints/5iter-64ch/stage3/epoch_249.pth')
    
    #512:32.09/0.922;  #new mask: 30.8/0.922;
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/STFormer/stformer_variable_cr_color.py') 
    # parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/stformer/stformer_base_mid_color.pth')


    #256:31.26/0.896; 512:14.32/0.249;  
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/RevSCI/revsci_variable_cr_color.py') 
    # parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/revsci/revsci.pth')

    #256:30.65/0.883; 512:16.89/0.343;  
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/BIRNAT/birant_variable_cr_color.py') 
    # parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/birnat/birnat.pth')

    
    #512: 21.03/0.527;   
    #parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/EfficientSCI/efficientsci_variable_cr_color.py')
    #parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/efficientsci/efficientsci_base_mid_color.pth')

    #512:        ;  
    # parser.add_argument("--config",type=str, default='/home/yutang/Documents/sci/DADUN/configs/EfficientSCI_plus_plus/efficientsci_plus_plus_variable_cr_color.py') 
    # parser.add_argument("--weights",type=str, default='/home/yutang/Documents/sci/DADUN/checkpoints/efficientsci_plus_plus_mid_color.pth')


    parser.add_argument("--work_dir",type=str,default=None)
    parser.add_argument("--device",type=str,default="cuda:3")
    parser.add_argument("--local_rank",default=-1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    local_rank = int(args.local_rank) 

    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0])


    log_dir = osp.join(args.work_dir,"log")
    train_image_save_dir = osp.join(args.work_dir,"test_images")


    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    if not osp.exists(train_image_save_dir):
        os.makedirs(train_image_save_dir)


    logger = Logger(log_dir)

    rank = 0 
    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    
    device = args.device
    model = build_model(cfg.model).to(device)
    logger.info("Load pre_train model...")
    resume_dict = torch.load(args.weights)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict,strict=True)

    if rank==0:
        logger.info('GPU info:\n' 
                + dash_line + 
                env_info + '\n' +
                dash_line)
        logger.info('cfg info:\n'
                + dash_line + 
                json.dumps(cfg, indent=4)+'\n'+
                dash_line) 
        # logger.info('Model info:\n'
        #         + dash_line + 
        #         str(model)+'\n'+
        #         dash_line)
    mask,mask_s = generate_cr_masks(cfg.test_data.mask_path,cfg.test_data.mask_shape)


    test_data = build_dataset(cfg.test_data,{"mask":mask})


    psnr_dict,ssim_dict = eval_psnr_ssim(model,test_data,mask,mask_s,args, cfg)

    psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
    ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
    logger.info("Mean PSNR: \n{}.\n".format(psnr_str))
    logger.info("Mean SSIM: \n{}.\n".format(ssim_str))

if __name__ == '__main__':
    main()


