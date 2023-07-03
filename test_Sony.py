import random
import torch
from torch.utils.data import DataLoader
import modelSID
import glob
import os
import numpy as np
import dataset
from PIL import Image

PATH = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/models/checkpoint_sony_e4000.pth'

input_dir = 'Seeing-in-the-Dark-Pytorch/Sony/short'
gt_dir = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/Sony/long'
result_dir = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/test_result/'
gt_fns = glob.glob(gt_dir + '1*.ARW')

train_fns = []

for i in gt_fns:
    _, filename = os.path.split(i)
    train = glob.glob(input_dir+filename[0:5]+'*') #return a list
    train_len = len(train)
    if train_len == 0:
        train_fns.append([])
    else:
        image = train[np.random.randint(0, train_len)]
        train_fns.append(image)

sonyTestset = dataset.LowLightSonyDataset(gt_fns, train_fns)
#sonyDataset.transform(sonyDataset.gt_list, sonyDataset.train_list)
print(len(sonyTestset))
dataloader_test = DataLoader(sonyTestset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True) 

def run_test(model, dataloader_test, images, PATH):
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(PATH))
        count = 0
        for image_num, low in enumerate(dataloader_test, 0):
            low = low.to(next(model.parameters()).device)
            low = low[1]
            gt = low[0]
            outputs = model(low)
            output = outputs.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output,0), 1)
            #gt = gt.permute(0, 2, 3, 1).cpu()
            #print(gt.shape, output.shape)
            temp = (output[0,:,:,:])
            Image.fromarray((temp*255).astype('uint8')).save(result_dir + f'{count:05}_00_train_.jpg')
            count+=1
            
            

model_ = modelSID()
