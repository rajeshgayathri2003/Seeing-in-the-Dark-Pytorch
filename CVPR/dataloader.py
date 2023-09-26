import cv2
import numpy as np
from patchify import patchify
import os
from torch.utils.data import DataLoader, Dataset
import gc
from PIL import Image
import torch
import imageio.v2 as imageio
from tqdm import tqdm

dir = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/well_lit/png/cam1/hist/'


       
class LowLightRGB(Dataset):
    def __init__(self, gt_list, in_list,dir=None, patchify= False, patchsize=None):
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
        
    def define_weights(self, num):
        weights = np.float32((np.logspace(0,num,127, endpoint=True, base=10.0)))
        weights = weights/np.max(weights)
        weights = np.flipud(weights).copy()    
        return weights

    def get_na(self, bins,weights,img_loww,amp=1.0):
        #print(img_loww.shape)
        #print(len(bins))
        H,W, _ = img_loww.shape
        arr = img_loww*1
        #print(len(weights))
        selection_dict = {weights[0]: (bins[0]<=arr)&(arr<bins[1])}
        #print(selection_dict)
        for ii in range(1,len(weights)):
            selection_dict[weights[ii]] = (bins[ii]<=arr)&(arr<bins[ii+1])
        #print(len(selection_dict))
        mask = np.select(condlist=selection_dict.values(), choicelist=selection_dict.keys())
        #print(img_loww)
        mask_sum1 = np.sum(mask,dtype=np.float64)
        
        na1 = np.float32(np.float64(mask_sum1*0.01*amp)/np.sum(img_loww*mask,dtype=np.float64))
        #print(na1)
        if na1>250.0:
            na1 = np.float32(250.0)
        if na1<1.0:
            na1 = np.float32(1.0)
        
        selection_dict.clear()

        return na1
    
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
        if gt_list == None:
            count = 0
            for i in range(len(in_list)):
                shortexposure = in_list[i]
                in_arr = Image.open(shortexposure)
                #ratio = self.find_ratio(shortexposure)
                ratio = 200
                in_arr = self.patchify_img(in_arr, 256)
                in_img = torch.from_numpy((in_arr/255)*ratio).permute(0, 3, 1, 2)
                self.in_files.append(in_img)
                self.gt_files.append([])
                
        else:
            count = 0
            for i in range(len(gt_list)):
                count+=1
                #if (count%20 ==0):
                    #print(count)
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
            gc.collect()
            print("DONE")
                
    def read_image(self, gt_list, in_list):
        if gt_list ==  None:
            count = 0
            bins = np.float32((np.logspace(0,8,128, endpoint=True, base=2.0)-1))/255.0
            weights5 = self.define_weights(5)
            for i in tqdm(range(len(in_list))):
                count+=1
                if (count%20 ==0):
                    print(count)
                in_image = in_list[i]
                in_arr = cv2.imread(in_image)
                in_arr = cv2.cvtColor(in_arr, cv2.COLOR_BGR2RGB)
                #print(in_arr.shape)
                na5 = self.get_na(bins,weights5,(np.array(in_arr)/255))
                print(na5)
                in_arr = cv2.resize(in_arr, (512, 512), interpolation=cv2.INTER_LINEAR)
                in_img = torch.from_numpy((np.array(in_arr)/255)*na5).permute(2, 0, 1).float()
                self.in_files.append(in_img)
                self.gt_files.append([])
        else:
            count = 0
            bins = np.float32((np.logspace(0,8,128, endpoint=True, base=2.0)-1))/255.0
            weights5 = self.define_weights(5)
            for i in range(len(gt_list)):
                count+=1
                #if (count%20 ==0):
                #    print(count)
                gc.collect()    
                image = gt_list[i]
                in_image = in_list[i]
                
                img_arr = imageio.imread(self.dir+image)
                in_arr = imageio.imread(in_image)
                
                na5 = self.get_na(bins,weights5,(np.array(in_arr)/255))
                #ratio = self.find_ratio(in_image)
                in_arr = cv2.resize(in_arr, (512, 512), interpolation=cv2.INTER_LINEAR)
                img_arr = cv2.resize(img_arr, (512, 512), interpolation=cv2.INTER_LINEAR)
                
                #Image.fromarray(img_arr).save('./new/method2{}.jpg'.format(i))
                img = torch.from_numpy(np.array(img_arr)/255).permute(2, 0, 1).float()
                in_img = torch.from_numpy((np.array(in_arr)/255)*na5).permute(2, 0, 1).float()
                self.gt_files.append(img)
                self.in_files.append(in_img)
                gc.collect()
            print("DONE")

            
    def __len__(self):
        return len(self.gt_list)
    
    def __getitem__(self, index):
        in_item = self.in_list[index]
        try:
            gt_item = self.gt_list[index]
        except Exception as e:
            gt_item = np.zeros_like(in_item)
        return gt_item, in_item
       

def Main():
    gt_dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/well_lit/png/cam1/hist/"
    exposure = [25, 50, 100, 200, 250]

    file_list = os.listdir(gt_dir)
    file_list = file_list[:152]
    in_list = []
    for i in range(len(file_list)):
        _, filename = os.path.split(file_list[i])
        #img_exposure = np.random.randint(0, len(exposure))
        img_exposure = 0
        in_dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/low_light/1_{}/png/cam1/adobe/".format(exposure[img_exposure])
        
        in_file = in_dir+filename
        with open("exposure.txt", "a") as f:
            f.write(filename+' '+str(img_exposure)+ '\n')
        #saved = cv2.imread(in_file)
        #cv2.imwrite("/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/presentation/jpg/{}.jpg".format(i), saved*exposure[img_exposure] )
        in_list.append(in_file)
        
    return file_list, in_list

def train():
    gt_fns, train_fns_new = Main()

    outdoorDataset = LowLightRGB(gt_fns, train_fns_new, dir)
    dataloader_train = DataLoader(outdoorDataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True) 
    print("REACHED")
    return dataloader_train

def test():
    test_dir = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/low_light/1_100/png/cam1/adobe/'

    file_list = os.listdir(test_dir)
    file_list_full  = [test_dir+x for x in file_list]

    gt_list = None
    outdoorDataset = LowLightRGB(gt_list, file_list_full, dir)
    
if __name__ == "__main__":
    train()
