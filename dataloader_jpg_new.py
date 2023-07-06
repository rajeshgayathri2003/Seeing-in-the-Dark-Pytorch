import cv2
import numpy as np
from patchify import patchify
import os
from torch.utils.data import DataLoader, Dataset
import gc
from PIL import Image
import torch
import imageio.v2 as imageio

dir = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/well_lit/png/cam1/hist/'

       
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
            self.read_image_patchify(gt_list, in_list)
        else:
            self.read_image(gt_list, in_list)
        self.gt_list = self.gt_files
        self.in_list = self.in_files
            
        #print(len(gt_list), len(in_list))
        
    def find_ratio(self, in_image):
        lst = in_image.split("/")
        val = lst[7][2:]
        return int(val)
          
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
    
    def read_image_patchify(self, gt_list, in_list):
        count = 0
        for i in range(len(gt_list)):
            count+=1
            if (count%20 ==0):
                print(count)
                gc.collect()
                
            _, gt_file = os.path.split(gt_list[i])
            gt_arr = Image.open(self.dir+gt_file)
            gt_arr = self.patchify_img(gt_arr, 256)
            
            shortexposure = in_list[i]
            in_arr = Image.open(shortexposure)
            ratio = self.find_ratio(shortexposure)
            in_arr = self.patchify_img(in_arr, 256)
            
            #print(gt_arr[0,:,:,:])
            #Image.fromarray(gt_arr[0,:,:,:]).save('./new/proper{}.jpg'.format(i))
            gt_img = torch.from_numpy(gt_arr/255).permute(0, 3, 1, 2)
            in_img = torch.from_numpy((in_arr/255)*ratio).permute(0, 3, 1, 2)
            self.gt_files.append(gt_img)
            self.in_files.append(in_img)
        print("DONE")
            
    def read_image(self, gt_list, in_list):
        count = 0
        for i in range(len(gt_list)):
            count+=1
            if (count%20 ==0):
                print(count)
                
            image = gt_list[i]
            in_image = in_list[i]
            
            img_arr = imageio.imread(self.dir+image)
            in_arr = imageio.imread(in_image)
            
            ratio = self.find_ratio(in_image)
            in_arr = cv2.resize(in_arr, (512, 512), interpolation=cv2.INTER_LINEAR)
            img_arr = cv2.resize(img_arr, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            
            #Image.fromarray(img_arr).save('./new/method2{}.jpg'.format(i))
            img = torch.from_numpy(np.array(img_arr)/255).permute(2, 0, 1).float()
            in_img = torch.from_numpy((np.array(in_arr)/255)*ratio).permute(2, 0, 1).float()
            self.gt_files.append(img)
            self.in_files.append(in_img)
        print("DONE")

            
    def __len__(self):
        return len(self.gt_list)
    
    def __getitem__(self, index):
        gt_item = self.gt_list[index]
        in_item = self.in_list[index]
        return gt_item, in_item
       

def Main():
    gt_dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/well_lit/png/cam1/hist/"
    exposure = [25, 50, 100, 200]

    file_list = os.listdir(gt_dir)
    in_list = []
    for i in range(len(file_list)):
        _, filename = os.path.split(file_list[i])
        img_exposure = np.random.randint(0, len(exposure))
        in_dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/low_light/1_{}/png/cam1/adobe/".format(exposure[img_exposure])
        in_file = in_dir+filename
        in_list.append(in_file)
        
    return file_list, in_list


gt_fns, train_fns_new = Main()

outdoorDataset = LowLightRGB(gt_fns, train_fns_new, dir)
dataloader_train = DataLoader(outdoorDataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True) 
print("REACHED")
