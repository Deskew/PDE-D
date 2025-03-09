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
    parser.add_argument("--config",type=str, default='./configs/MIXST_SCI/mixst_sci_variable_cr.py')
    parser.add_argument("--weights",type=str, default='./checkpoints/pde-d/PDE-D-5st_gray.pth')
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


