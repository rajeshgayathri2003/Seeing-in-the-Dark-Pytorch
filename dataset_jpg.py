import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset
import rawpy
from PIL import Image
import gc

def calcRatio(gt_img, in_img):
    pass

def transform(input_patch, gt_patch):
    if np.random.randint(2,size=1)[0] == 1:  # random flip 
        input_patch = np.flip(input_patch, axis=1)
        gt_patch = np.flip(gt_patch, axis=1)
    if np.random.randint(2,size=1)[0] == 1: 
        input_patch = np.flip(input_patch, axis=2)
        gt_patch = np.flip(gt_patch, axis=2)
    if np.random.randint(2,size=1)[0] == 1:  # random transpose 
        input_patch = np.transpose(input_patch, (0,2,1,3))
        gt_patch = np.transpose(gt_patch, (0,2,1,3))
        
    return input_patch, gt_patch


def readimage(gt_list, in_list, patchsize):
    gt_list_images = []
    in_list_images = []
    count = 0
    for i in range(len(gt_list)):
        count+=1
        if (count%20 == 0):
            print(count)
            
        _, gt_file = os.path.split(gt_list[i])
        _, in_file = os.path.split(in_list[i])
        gt_image = Image.open(gt_list[i])
        ##choose whether to keep this or not
        ##gt_image = gt_image.resize(512, 512)
        ##
        gt_image_array = Image.fromarray(gt_image, np.float32)
        gt_image_array = np.expand_dims(gt_image_array/ 65535.0, axis = 0)
        
        _, h, w,_ = gt_image_array.shape
        
        short_images = in_list[i]
        in_image = Image.open(short_images)
        ##choose whether to keep this or not
        ##in_image = in_image.resize(512, 512)
        ##
        in_image_array = Image.fromarray(in_image, np.float32)
        #subtracting black levels
        in_image_array = np.expand_dims((np.maximum(in_image_array-512, 0)/ (16383-512)), axis=0)
        
        H = in_image_array.shape[1]
        W = in_image_array.shape[2]
        
        xx = np.random.randint(0, W-patchsize)
        yy = np.random.randint(0, H-patchsize)
                
        img_patch = in_image_array[:,yy:yy+patchsize, xx: xx+patchsize, :]
        gt_patch = gt_image_array[:,yy*2:yy*2+patchsize*2, xx*2: xx*2+patchsize*2, :]
        
        img_patch, gt_patch = transform(img_patch, gt_patch)
        
        input_patch = np.minimum(img_patch,1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        
        gc.collect()  
        gt_patch = np.squeeze(gt_patch)
        input_patch = np.squeeze(input_patch)
        gt_list_images.append(gt_patch)
        in_list_images.append(input_patch)
           
    return gt_list_images, in_list_images

class LowLightSonyDataset(Dataset):
    def __init__(self, gt_list, in_list):
        print("..........Loading Train Images..........")
        self.gt_list, self.in_list = readimage(gt_list, in_list, 512)
        
    def __len__(self):
        return len(self.gt_list)
    
    def __getitem__(self, idx):
        img_gtt = self.gt_list[idx]
        img_loww = self.in_list[idx]
        return img_gtt, img_loww    
        