import os, time
import torch
import torch.optim as optim
import numpy as np
import rawpy
import glob

from modelSID import SeeingInDark_RGB
import dataloader_jpg_new 
from PIL import Image

input_dir = '/home/atreyee/Gayathri/Learning_to_See_in_the_Dark_PyTorch/dataset/Sony/short/'
gt_dir = '/home/atreyee/Gayathri/Learning_to_See_in_the_Dark_PyTorch/dataset/Sony/long/'
checkpoint_dir = './result_Sony/'
result_dir = './result_Sony/'
dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/rec_outdoor2/dataset_may15_part0_rect/cam1/RGB/"

ps = 512 #patchsize
save_freq = 500

#setting gpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def G_loss(out_image, gt_image):
    return torch.mean(torch.abs(out_image-gt_image))


train_loader = dataloader_jpg_new.train()
  
'''       
for i, data in enumerate(train_loader, 0):
    low = data[1]
    gt = data[0]
    print(low.shape, gt.shape)
'''

def train(lastepoch, savefrquency):
    result_dir = './result_jpg/'
    model_dir = './models/jpg/'
    #enable adam
    PATH = model_dir+'model.pt'
    learning_rate = 1e-4
    _model = SeeingInDark_RGB().to(device)
    _model._initialise_weights()
    optimizer = optim.Adam(_model.parameters(), lr = learning_rate)
    for epoch in range(lastepoch, 4001):
        torch.cuda.empty_cache()
        if (epoch%50 == 0):
            print("================EPOCH {}================".format(epoch))
        if epoch>2000:
            for i in optimizer.param_groups: 
                i['lr'] = 1e-5
        count = 0
        for i, data in enumerate(train_loader, 0):
            count+=1
            low = data[1]
            gt = data[0]
            #print(low.shape, gt.shape)
            #gt_test = gt.numpy()
            #Image.fromarray((gt_test[0,:,:,:]*255).astype('uint8')).save('./new/{}.jpg'.format(i))
            #print(low.shape, gt.shape)
            flag = 0
            if isinstance(low, type([])) or isinstance(gt, type([])):
                pass
            else: 
                low =  low.to(device)
                gt_new = gt.to(device)
                optimizer.zero_grad()
                outputs = _model(low)
                #print(outputs.shape, gt_new.shape)
                loss = G_loss(outputs, gt_new)
                loss.backward()
                optimizer.step()
                flag = 1 
            if (count%50 == 0):
                torch.save({
                'epoch': epoch,
                'model_state_dict': _model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)
                print(count)
                
            if (epoch%savefrquency == 0 and flag):
                epoch_result_dir = result_dir+f'{epoch:04}/'
                #print('Hi')
                if not os.path.isdir(epoch_result_dir):
                    os.makedirs(epoch_result_dir)
                    #print('Hi')
                    
                output = outputs.permute(0, 2, 3, 1).cpu().data.numpy()
                output = np.minimum(np.maximum(output,0), 1)
                gt = gt.permute(0, 2, 3, 1).cpu()
                #print(gt.shape, output.shape)
                #print(gt.shape, output.shape)
                temp = np.concatenate((gt[0,:,:,:], output[0,:,:,:]),axis=1)
                Image.fromarray((temp*255).astype('uint8')).save(epoch_result_dir + f'{count:05}_00_train_.jpg')
                torch.save(_model.state_dict(), model_dir+'checkpoint_sony_e%04d.pth'%epoch)
                
   
lastepoch = 0 
train(lastepoch, 100)