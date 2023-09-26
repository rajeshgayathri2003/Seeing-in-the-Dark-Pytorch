import rawpy
import numpy as np
from PIL import Image

png_dataset = '/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/png_from_raw'

def createPNGdataset_gt(image_list):
    for i in range(len(image_list)):
        raw = rawpy.imread(image_list[i])
        img_gt = raw.postprocess(use_camera_wb = True,half_size=False, no_auto_bright=True, output_bps=16).copy()
        new_image = Image.fromarray(img_gt.astype(np.uint8))
        new_image.save(png_dataset+'/gt/%04d'%i)
        
def createPNGdataset_in(image_list):
    for i in range(len(image_list)):
        raw = rawpy.imread(image_list[i])
        img_gt = raw.postprocess(use_camera_wb = True,half_size=False, no_auto_bright=True, output_bps=16).copy()
        new_image = Image.fromarray(img_gt.astype(np.uint8))
        new_image.save(png_dataset+'/in/%04d'%i)
    
        
        

