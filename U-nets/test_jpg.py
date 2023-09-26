import random
import torch
from torch.utils.data import DataLoader
import modelSID
import glob
import os
import numpy as np
import dataset
from PIL import Image
from dataloader_jpg_new import LowLightRGB

PATH = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/models/jpg/checkpoint_sony_e1200.pth'

test_dir = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/test/'

file_list = os.listdir(test_dir)
file_list_full  = [test_dir+x for x in file_list]

gt_list = None

oxfordTest = LowLightRGB(gt_list, file_list_full)
dataloader_test = DataLoader(oxfordTest, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

def run_test(model, dataloader_test,PATH):
    result_dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/test_jpg_result/"
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(PATH))
        count = 0
        for image_num, low in enumerate(dataloader_test, 0):
            low = low[1]
            gt = low[0]
            #print(low.shape)
            low =  low.to(next(model.parameters()).device)
            #gt_new = gt.permute(0,3,1,2).to(next(model.parameters()).device)
            low = low.to(next(model.parameters()).device)
            outputs = model(low)
            output = outputs.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output,0), 1)
            #gt = gt.permute(0, 2, 3, 1).cpu()
            #print(gt.shape, output.shape)
            temp = (output[0,:,:,:])
            Image.fromarray((temp*255).astype('uint8')).save(result_dir + f'{count:05}_00_test_.jpg')
            count+=1
            
            

model_ = modelSID.SeeingInDark_RGB()
run_test(model_, dataloader_test, PATH )
print("DONE")
