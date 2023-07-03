import dataset_jpg
import glob
import os
import numpy as np
from torch.utils.data import DataLoader

input_dir = ""
gt_dir = ""

gt_fns=[]
train_fns=[]

dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/rec_outdoor2/dataset_may15_part0_rect/cam1/RGB/"

for i in gt_fns:
    _, filename = os.path.split(i)
    train = glob.glob(input_dir+filename[0:5]+'*') #return a list
    train_len = len(train)
    if train_len == 0:
        train_fns.append([])
    else:
        image = train[np.random.randint(0, train_len)]
        train_fns.append(image)
        

sonyDataset = dataset_jpg.LowLightSonyDataset(gt_fns, train_fns)
#sonyDataset.transform(sonyDataset.gt_list, sonyDataset.train_list)
dataloader_train = DataLoader(sonyDataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True) 