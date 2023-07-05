import cv2
import numpy as np
from patchify import patchify
import os
from torch.utils.data import DataLoader, Dataset
import gc
from PIL import Image
import torch
import imageio.v2 as imageio

dir = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/rec_outdoor2/dataset_may15_part0_rect/cam1/RGB/'

        
class LowLightRGB(Dataset):
    def __init__(self, gt_list, in_list,dir, patchify= False, patchsize=None):
        self.gt_list = gt_list
        self.in_list = in_list
        self.dir = dir
        self.patchsize = patchsize
        self.gt_files = []
        self.in_files = []
        print("..........Loading Training Data..........")
        if patchify:
            self.read_image_patchify(gt_list)
        else:
            self.read_image(gt_list)
        
    def patchify_img(self, image, patch_size):
        #print(type(image))
        image_array = np.asarray(image)
        size_x = (image_array.shape[0]//patch_size)*patch_size
        size_y = (image_array.shape[1]//patch_size)*patch_size
        
        #crop original image to required size
        image = image_array[:size_x, :size_y, :]
        patch_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
        
        patches = []
        
        for j in range(patch_img.shape[0]):
            for k in range(patch_img.shape[1]):
                single_patch_image = patch_img[j,k]
                
                patches.append(np.squeeze(single_patch_image))       
        return np.vstack([np.expand_dims(x, 0) for x in patches])
    
    def read_image_patchify(self, gt_list):
        for i in range(len(gt_list)):
            _, gt_file = os.path.split(gt_list[i])
            gt_arr = Image.open(self.dir+gt_file)
            gt_arr = self.patchify_img(gt_arr, 256)
            #print(gt_arr[0,:,:,:])
            #Image.fromarray(gt_arr[0,:,:,:]).save('./new/proper{}.jpg'.format(i))
            gt_img = torch.from_numpy(gt_arr/255).permute(0, 3, 1, 2)
            self.gt_files.append(gt_img)
        print("DONE")
            
    def read_image(self, gt_list):
        for i in range(len(gt_list)):
            image = gt_list[i]
            img_arr = imageio.imread(self.dir+image)
            
            img_arr = cv2.resize(img_arr, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            #Image.fromarray(img_arr).save('./new/method2{}.jpg'.format(i))
            img = torch.from_numpy(np.array(img_arr)/255).permute(2, 0, 1).unsqueeze(0)
            self.gt_files.append(img)
        print("DONE")

            
    def __len__(self, img_list):
        return len(img_list)
    
    def __getitem__(self, index):
        gt_item = self.gt_list[index]
        in_item = self.in_list[index]
        return gt_item, in_item
       

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

gt_fns, train_fns_new = Main([],[], [], dir)
LowLightRGB(gt_fns, train_fns_new, dir)
#read_image(gt_fns)