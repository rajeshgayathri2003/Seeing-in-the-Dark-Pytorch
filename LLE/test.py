import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from tqdm import tqdm 
import dataloader
import numpy as np
import unet
import utils
from torch.optim import SGD, Adam
from PIL import Image
import yaml

def make_dataloader():
    gt_dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/well_lit/png/cam1/hist/"
    in_dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/low_light/1_{}/png/cam1/adobe/".format(200)
    exposure = [25, 50, 100, 200, 250]

    file_list = os.listdir(gt_dir)
    file_list = file_list[141:200]
    in_list = []
    for i in range(len(file_list)):
        _, filename = os.path.split(file_list[i])
        #img_exposure = np.random.randint(0, len(exposure))
        in_dir = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/dataset/low_light/1_{}/png/cam1/adobe/".format(200)
        
        in_file = in_dir+filename
        #saved = cv2.imread(in_file)
        #cv2.imwrite("/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/presentation/jpg/{}.jpg".format(i), saved*exposure[img_exposure] )
        in_list.append(in_file)
        
    outdoorDataset = dataloader.LowLightRGB(gt_list=file_list, in_dir=in_dir, gt_dir = gt_dir, patchify= True, patchsize=384, cache=True)
    #outdoorDataset = dataloader.LowLightRGB(file_list, in_list, dir=gt_dir)
    dataloader_train = DataLoader(outdoorDataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        
    return dataloader_train

def G_loss(out_image, gt_image):
    return torch.mean(torch.abs(out_image-gt_image))

def prepare_testing(optim_spec, lr):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    net = unet.UNet().to(device)
    optim_dict = {'Adam': Adam, 'SGD':SGD}
    optim_spec = optim_dict[optim_spec]
    optimizer = utils.make_optimizer(optimizer_spec=optim_spec, model=net, lr=lr)
    #lr_scheduler = MultiStepLR(optimizer=optimizer, milestones= milestones,gamma=gamma)
    '''
    if loss == None:
        loss_fn = nn.MSELoss()
    elif loss == 'l1':
        loss_fn = nn.L1Loss()
    elif loss == 'Gloss':
        loss_fn = G_loss
    '''
    return net, optimizer, device

def testing(train_loader, model, device, optimizer, epoch_result_dir, PATH, load = False):
    model.train()
    count = 0
    loss_list = []
    loss_iter_list = []
    #PATH = PATH + '00010_model.pt00020_model.pt00025_model.pt'
    #save_csv_files = "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/LLE/csv/"
    lastepoch = 0
    #optimizer.load_state_dict(torch.load(PATH)['optimizer_state_dict'])
    #lastepoch = torch.load(PATH)['epoch']
    #loss = torch.load(PATH)['loss']
    with open('config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    if load:
        lastepoch = torch.load(PATH+'model.pt')['epoch']
        
        loss = torch.load(PATH+'model.pt')['loss']
        optimizer.load_state_dict(torch.load(PATH+'model.pt')['optimizer_state_dict'])
        model.load_state_dict(torch.load(PATH+'model.pt')['model_state_dict'])
        
    
    for i, batch in enumerate(train_loader, 0):
        with torch.no_grad():
                low = batch[1]
                gt_first = batch[0]
                low = low.squeeze()
                gt = gt_first.squeeze()
                
                #normalisation
                '''
                sub = 0.5
                div = -0.5
                sub = torch.FloatTensor([sub]).view(1, -1, 1, 1)
                div = torch.FloatTensor([div]).view(1, -1, 1, 1)
                low = (low-sub)/(div)
                gt = (gt-sub)/(div)
                '''
                    
                low = low.to(device).float()

                gt = gt.to(device).float()
                    
                optimizer.zero_grad()
                #print(type(low))
                outputs = model(low)
                    
                #loss = loss_fn(outputs, gt)
                outputs = outputs.permute(0, 2, 3, 1).cpu()
                output = outputs.data.numpy()
                output_ = np.minimum(np.maximum(output,0), 1) #clamping between 0 and 1
                
                gt = gt.permute(0, 2, 3, 1).cpu()
                
                
                
                psnr_train = utils.Averager()
                #loss_train = utils.Averager()
                #loss_train.add(loss.item())
                
                psnr_val = utils.calc_psnr(outputs, gt)
                
                psnr_train.add(psnr_val.item())
                val_write = psnr_train.item()
                
                
                config['best_val'] = (val_write)
                with open('config.yaml', 'w') as f:
                    yaml.dump(config, f, sort_keys=False)
                    
                    
                
                directory_path = epoch_result_dir+f'{epoch:04}/'
                temp = np.concatenate((gt[0,:,:,:], output_[0,:,:,:]),axis=1)
                if os.path.exists(directory_path) and os.path.isdir(directory_path):
                    Image.fromarray((temp*255).astype('uint8')).save(epoch_result_dir+f'{epoch:04}/' + f'{i:05}_00_train_.jpg')
                else:
                    os.makedirs(directory_path)
                    Image.fromarray((temp*255).astype('uint8')).save(epoch_result_dir+f'{epoch:04}/' + f'{i:05}_00_train_.jpg')
                
        
        

                
        
    #np.savetxt(os.path.join(save_csv_files,'loss_curve_1.csv'),[p for p in zip(loss_iter_list,loss_list)],delimiter=',',fmt='%s')  
            
train_loader = make_dataloader()
with open('config_test.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
loss = config['loss']
optimizer = config['optimizer']
lr = float(config['lr'])
epoch = int(config['epoch'])
milestones = config['milestones']

milestones = [int(milestone) for milestone in milestones]
gamma = float(config['gamma'])

net, optimizer, device= prepare_testing(optimizer, lr)
testing(train_loader=train_loader, model= net, device=device, optimizer=optimizer, epoch_result_dir= "/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/LLE/test_results/", PATH="/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/LLE/models/", load=True)
