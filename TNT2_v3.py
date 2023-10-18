#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 07:11:07 2022

@author: yasinceran
"""

#pip install segmentation-models-pytorch
#pip install pytorch-lightning==1.5.4
#pip install numpy libtiff (conda install libtiff=4.2.0)
#pip install albumentations
import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from albumentations import (
    Resize,
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma    
)

from util import create_dir,to_numpy,resize_my_images,load_image,load_set,save_to_tif
import torch
from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler