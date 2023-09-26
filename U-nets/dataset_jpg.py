import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import gc

#Well lit cam1 hist
#low cam1 adobe

gt_dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/well_lit/png/cam1/hist/"
exposure = [25, 50, 100, 200]

file_list = os.listdir(gt_dir)
in_list = []
for i in range(len(file_list)):
    _, filename = os.path.split(file_list[i])
    img_exposure = random.randint(0, len(exposure)-1)
    in_dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/low_light/1_{}/png/cam1/adobe/".format(exposure[img_exposure])
    in_file = in_dir+filename
    in_list.append(in_file)
    


'''

def Main(gt_fns, train_fns, train_fns_new, dir):
    file_list = os.listdir(dir)
    dict_img = {}
    for i in file_list:
        vals = i.split('_')
        if vals[0] not in dict_img:
            dict_img[vals[0]] = [i]
        else:
            dict_img[vals[0]].append(i)
            
    for key in dict_img:
        val = dict_img[key]
        max_ = None
        gt = None
        for j in val:
            exposure = j.split('_')[5]
            if max_ == None:
                max_ = float(exposure)
                gt = j
            else:
                if float(exposure)> max_:
                    max_ = float(exposure)
                    gt = j
                    
        gt_fns.append(gt)
        dict_img[key].remove(gt)
        train_fns.append(dict_img[key])
        
    #print(gt_fns)
    #print(train_fns)

    for i in train_fns:
        train_len = len(i)
        image = i[np.random.randint(0, train_len)]
        train_fns_new.append(image)
        
    return gt_fns, train_fns_new
    
    
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
    dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/rec_outdoor2/dataset_may15_part0_rect/cam1/RGB/"
    gt_list_images = []
    in_list_images = []
    count = 0
    for i in range(len(gt_list)):
        count+=1
        if (count%20 == 0):
            print(count)
        #print(in_list[i])    
        _, gt_file = os.path.split(gt_list[i])
        _, in_file = os.path.split(in_list[i])
        gt_image = Image.open(dir+gt_list[i])
        gt_image_array = np.asarray(gt_image, np.float32)
        #print(gt_image_array.dtype)
        gt_image_array = (np.float32(gt_image_array/ 65535.0))
        
        #gt_image_array_new = (gt_image_array*255).astype(np.uint8)
        #print(gt_image_array_new)
        #Image.fromarray((gt_image_array[:,:,:]*255).astype(np.uint8)).save('./new/check_again2{}.jpg'.format(i))
        gt_image_array = np.expand_dims(gt_image_array, axis = 0)
        _, h, w,_ = gt_image_array.shape
        
        short_images = in_list[i]
        in_image = Image.open(dir+short_images)
        ##choose whether to keep this or not
        ##in_image = in_image.resize(512, 512)
        ##
        in_image_array = np.asarray(in_image, np.float32)
        #subtracting black levels
        in_image_array = np.expand_dims((np.maximum(in_image_array-512, 0)/ (16383-512)), axis=0)
        
        H = in_image_array.shape[1]
        W = in_image_array.shape[2]
        
        #xx = np.random.randint(0, W-patchsize)
        #yy = np.random.randint(0, H-patchsize)
        
        #img_patch = in_image_array
        #gt_patch = gt_image_array        
        img_patch = in_image_array[:,:patchsize, :patchsize, :]
        gt_patch = gt_image_array[:,:patchsize*2, :patchsize*2, :]
        img_patch, gt_patch = transform(img_patch, gt_patch)
        
        input_patch = np.minimum(img_patch,1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        
        gc.collect()  
        gt_patch = np.squeeze(gt_patch)
        #Image.fromarray((gt_patch[:,:,:]*255).astype('uint8')).save('./new/{}.jpg'.format(i))
        input_patch = np.squeeze(input_patch)
        #print(gt_patch.shape, input_patch.shape)
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
'''