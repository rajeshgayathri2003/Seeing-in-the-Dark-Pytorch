from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam
import torch
import os
import torch

def make_optimizer(optimizer_spec, model, lr, PATH = None, load = False):
    optimizer = optimizer_spec(model.parameters(), lr = lr)
    if load:
        optimizer.load_state_dict(torch.load(PATH)['optimizer_state_dict'])
        
    return optimizer
        
_log_path = None

def set_log_path(save_path):
    global _log_path
    _log_path = save_path
    
def log(obj, filename='log.txt'):
    #prints the object passed in console
    print(obj)

    #prints into the log file
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)
            
def ensure_save_path(path, remove= True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove:
            #remove all except yaml file but the epoch models will be lost
            files  = [f for f in os.listdir(path) if not (f.endswith('.yaml') or f.endswith('epoch_last.pth'))]
            for file in files:
                os.remove(os.path.join(path, file))
    else:
        os.makedirs(path)


def set_save_path(save_path, remove = True):

    #ensure path exists by creating or removing if already existing
    ensure_save_path(save_path, remove = False)

    #setting the path to log file within save dir
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    #returning log func
    return log, writer

def calc_psnr(out, gt, rgb_range=1):
    diff = (out-gt)/rgb_range
    mse = torch.mean(torch.pow(diff, 2))
    return -10 * torch.log10(mse)

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v