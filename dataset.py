import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset
import rawpy
from PIL import Image
import gc

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

def ratio_find(in_fn, gt_fn):
    in_exposure = float(in_fn[9:-5])
    gt_exposure = float(gt_fn[9:-5])
    ratio = min(gt_exposure / in_exposure, 300)
    return ratio

def packraw(im):
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def readimage(gt_list, in_list, patchsize):
    gt_list_images = []
    in_list_images = []
    in_patch = []
    count = 0
    print(len(gt_list))
    for i in range(len(gt_list)):
        count+=1
        if (count%20 == 0):
            print(count)
        _, gt_file = os.path.split(gt_list[i])
        if (isinstance(in_list[i], type([])) ):
            pass
        else:
            _, in_file = os.path.split(in_list[i])
            ratio = ratio_find(in_file, gt_file)
            raw = rawpy.imread(gt_list[i])
            img_gt = raw.postprocess(use_camera_wb = True,half_size=False, no_auto_bright=True, output_bps=16).copy()
            img_gtt=np.expand_dims(np.float32(img_gt/65535.0), axis=0)
            _, h,w,_ = img_gtt.shape
            
            correct_dim_flag = False
            if h%32!=0:
                print('Correcting the 1st dimension.')
                h = (h//32)*32
                img_gtt = img_gtt[:h,:,:]
                correct_dim_flag = True
            
            if w%32!=0:
                print('Correcting the 2nd dimension.')
                w = (w//32)*32
                img_gtt = img_gtt[:,:w,:]
                correct_dim_flag = True
                
            short_images = in_list[i]
            #print(short_images)
            if len(short_images) == 0:
                input_patch = in_patch
            else:
                    
                #n_path = short_images[np.random.random_integers(0, len(short_images) - 1)]
                raw = rawpy.imread(short_images)
                #raw = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                img = raw.raw_image_visible.astype(np.float32).copy()
                img_loww = (np.maximum(img - 512,0)/ (16383 - 512))
                
                img = np.expand_dims(packraw(img_loww), axis=0) * ratio
                H = img.shape[1]
                W = img.shape[2]
                #scale_full = np.expand_dims(np.float32(raw / 65535.0), axis=0)
                xx = np.random.randint(0, W-patchsize)
                yy = np.random.randint(0, H-patchsize)
                
                img_patch = img[:,yy:yy+patchsize, xx: xx+patchsize, :]
                gt_patch = img_gtt[:,yy*2:yy*2+patchsize*2, xx*2: xx*2+patchsize*2, :]
                img_patch, gt_patch = transform(img_patch, gt_patch)
                raw.close()
                
                if correct_dim_flag:
                    img = img[:h,:w]        
                
                input_patch = np.minimum(img_patch,1.0)
                gt_patch = np.maximum(gt_patch, 0.0)
            
            gc.collect()  
            gt_patch = np.squeeze(gt_patch)
            input_patch = np.squeeze(input_patch)
            gt_list_images.append(gt_patch)
            in_list_images.append(input_patch)   
            raw.close()
    print(len(gt_list_images), len(in_list_images)) 
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