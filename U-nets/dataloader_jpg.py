import dataset_jpg
import glob
import os
import numpy as np
from torch.utils.data import DataLoader

input_dir = ""
gt_dir = ""

gt_fns=[]
train_fns=[]
train_fns_new = []

dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/rec_outdoor2/dataset_may15_part0_rect/cam1/RGB/"
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
   
#print(train_fns_new) 
'''
for i in gt_fns:
    _, filename = os.path.split(i)
    train = glob.glob(input_dir+filename[0:5]+'*') #return a list
    train_len = len(train)
    if train_len == 0:
        train_fns.append([])
    else:
        image = train[np.random.randint(0, train_len)]
        train_fns.append(image)
        
'''
#print(train_fns_new)
sonyDataset = dataset_jpg.LowLightSonyDataset(gt_fns, train_fns_new)
#sonyDataset.transform(sonyDataset.gt_list, sonyDataset.train_list)
dataloader_train = DataLoader(sonyDataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True) 
