import random
import torch
from torch.utils.data import Dataloader
import modelSID
import glob
import os
import numpy as np
import dataset

PATH = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/models/checkpoint_sony_e4000.pth'

input_dir = '/home/atreyee/Gayathri/pytorch-Learning-to-See-in-the-Dark/dataset/Sony/short/'
gt_dir = '/home/atreyee/Gayathri/pytorch-Learning-to-See-in-the-Dark/dataset/Sony/long/'
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
dataloader_train = DataLoader(sonyTestset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True) 

def run_test(model, dataloader_test, images, PATH):
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(PATH))
        for image_num, low in enumerate(dataloader_test):
            low = low.to(next(model.parameters()).device)
            
            

model_ = modelSID()
