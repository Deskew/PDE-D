from torch.utils.data import Dataset 
import numpy as np
import os
import os.path as osp
import cv2
from .pipelines import Compose
from .builder import DATASETS
from .pipelines.builder import build_pipeline

@DATASETS.register_module 
class VariableCrBayerData(Dataset):
    def __init__(self,data_root,*args,**kwargs):
        self.data_dir= data_root
        self.data_list = os.listdir(data_root)
        self.img_files = []
        
        self.mask = kwargs["mask"]
        self.ratio,self.mask_h,self.mask_w= self.mask.shape
        r = np.array([[1, 0], [0, 0]])
        g1 = np.array([[0, 1], [0, 0]])
        g2 = np.array([[0, 0], [1, 0]])
        b = np.array([[0, 0], [0, 1]])
        self.rgb2raw = np.zeros([3, self.mask_h, self.mask_w])
        self.rgb2raw[0, :, :] = np.tile(r, (self.mask_h // 2, self.mask_w // 2))
        self.rgb2raw[1, :, :] = np.tile(g1, (self.mask_h // 2, self.mask_w // 2)) + np.tile(g2, (
            self.mask_h // 2, self.mask_w // 2))
        self.rgb2raw[2, :, :] = np.tile(b, (self.mask_h // 2, self.mask_w // 2))

        #self.pipeline = Compose(kwargs["pipeline"])
        self.gene_meas = build_pipeline(kwargs["gene_meas"])

        for image_dir in os.listdir(data_root):
            test_data_path = osp.join(data_root,image_dir)
            data_path = os.listdir(test_data_path)
            data_path.sort()
            count = 0
            image_name_list = []
            for image_name in data_path:
                image_name_list.append(osp.join(test_data_path,image_name))
                if (count+1)%self.ratio==0:
                    self.img_files.append(image_name_list)
                    image_name_list = []
                count += 1
        
    def __getitem__(self,index):
        imgs = []
        for i,image_path in enumerate(self.img_files[index]):
            img = cv2.imread(image_path)
            imgs.append(img)
        gt,meas = self.gene_meas(imgs,self.mask,self.rgb2raw)
        return gt,meas 
       
    def __len__(self,):
        return len(self.img_files)