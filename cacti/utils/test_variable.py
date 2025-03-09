import os
import os.path as osp
from torch.utils.data.dataloader import DataLoader 
import torch 
from cacti.utils.utils import save_image,save_single_image
from cacti.utils.metrics import compare_psnr,compare_ssim, calculate_psnr, calculate_ssim
import numpy as np 
import einops 

def eval_psnr_ssim(model,test_data,mask,mask_s,args, cfg):
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]
    data_loader = DataLoader(test_data,1,shuffle=False,num_workers=4)
    cr, h, w = mask.shape
    #temp = test_data.img_files
    for iter,data in enumerate(data_loader):
        psnr,ssim = 0,0
        batch_output = []

        gt, meas = data# color:[1, 3, 8, 512, 512],[1, 512, 512]
        gt = gt[0].numpy() #color: (3, 8, 512, 512)
        #gt = einops.rearrange(gt,"cr c h w->c cr h w")
        if len(gt.shape)==4:
            c,cr,h, w = gt.shape
            gt = einops.rearrange(gt,"c cr h w->cr h w c")
        else:
            single_gt = gt

        meas = meas[0].float().to(args.device) #[512, 512]
        batch_size = 1#meas.shape[0]
        Phi = einops.repeat(mask,'cr h w->b cr h w',b=1)
        Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=1)
        
        # if cr > 8:
        #     phi = []
        #     Phi_s = []
        #     n = 0
        #     # Starting index for taking sub-elements
        #     start_idx = 0
        #     # Recursive function to take 8 sub-elements at a time or add elements from the previous time
        #     # Check if there are at least 8 sub-elements remaining
        #     while(True):
        #         if start_idx + 8 <= cr:
        #             sub_mask = mask[start_idx : start_idx + 8, :, :]
        #             sub_mask_s = np.sum(sub_mask, axis=0)
        #             sub_mask_s[sub_mask_s==0] = 1
        #             sub_phi_s = einops.repeat(sub_mask_s,'h w->b 1 h w',b=1)
        #             Phi_s.append(sub_phi_s)
        #             phi.append(Phi[:,start_idx : start_idx + 8, :, :])
        #             n+=1
        #             if start_idx + 8 == cr:
        #                 start_idx += 8
        #                 Phi = phi
        #                 break
        #             start_idx += 8
        #         else:
        #             # Calculate the number of elements to add from the previous time
        #             num_missing = 8 - (cr - start_idx)
        #             sub_mask = mask[start_idx-num_missing :, :, :]
        #             sub_mask_s = np.sum(sub_mask, axis=0)
        #             sub_mask_s[sub_mask_s==0] = 1
        #             sub_phi_s = einops.repeat(sub_mask_s,'h w->b 1 h w',b=1)
        #             Phi_s.append(sub_phi_s)
        #             Phi_s = np.concatenate(Phi_s)
        #             phi.append(Phi[:, start_idx-num_missing:, :, :])
        #             phi = np.concatenate(phi)
        #             n+=1
        #             Phi = phi
        #             break
        #         # 
            
        # else:
        #    Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=1)

        Phi = torch.from_numpy(Phi).to(args.device)
        Phi_s = torch.from_numpy(Phi_s).to(args.device)
        
        for ii in range(batch_size):
            
            single_gt = gt
            single_meas = meas.unsqueeze(0).unsqueeze(0)
            #single_meas = 8 * (single_meas/cr) 
            with torch.no_grad():
                # if cr > 8:
                #     for n_j in range(n):
                #         output = model(single_meas, Phi[n_j,:, :, :], Phi_s[n_j,:, :, :])
                #         if n_j + 1 == n and start_idx != cr:
                #             outputs = torch.cat((outputs, output[0][:, num_missing:, :, :]), dim=1)
                #         else:
                #             if n_j == 0:
                #                 outputs = output[0]
                #             else:
                #                 outputs = torch.cat((outputs, output[0]), dim=1)
                # else:
                #     outputs = model(single_meas, Phi, Phi_s)
                if cfg.model['type'] == 'DeSCI':
                    outputs = model(single_meas,cfg.sigma_list)
                else:
                    outputs = model(single_meas, Phi, Phi_s)
            if not isinstance(outputs,list):
                outputs = [outputs]
            output = outputs[-1][0].cpu().numpy() #c cr h w
            batch_output.append(output)
            for jj in range(cr):
                if output.shape[0]==3:
                    per_frame_out = output[:,jj]
                    per_frame_out = einops.rearrange(per_frame_out,"c h w->h w c")
                    psnr += calculate_psnr(per_frame_out*255, gt[jj]*255)
                    ssim += calculate_ssim(per_frame_out*255, gt[jj]*255)
                    # rgb2raw = test_data.rgb2raw
                    # per_frame_out = np.sum(per_frame_out*rgb2raw,axis=0)
                else:
                    per_frame_out = output[jj]
                    per_frame_gt = single_gt[jj, :, :] #cr h w c
                    psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                    ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
                
        psnr = psnr / (batch_size * cr)
        ssim = ssim / (batch_size * cr)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        out_list.append(np.array(batch_output))
        gt_list.append(gt)

    test_dir = osp.join(args.work_dir,"test_images")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)

    for i,name in enumerate(test_data.img_files):

        psnr_dict["batch_"+str(i)] = psnr_list[i]
        ssim_dict["batch_"+str(i)] = ssim_list[i]
        out = out_list[i]
        gt = gt_list[i]

        for j in range(out.shape[0]):
            if len(gt.shape)==4:
                save_single_image(out[j],test_dir,i,name="_rec")
            else:
                save_single_image(out,test_dir,i,name="_rec")
            
    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)
    return psnr_dict,ssim_dict