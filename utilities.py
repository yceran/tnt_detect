#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:06:44 2022

@author: yasinceran
"""

import matplotlib.pyplot as plt
import os 
from libtiff import TIFF
import numpy as np
from PIL import Image
#from libtiff import TIFF
import skimage.io
import glob
import cv2

def plot_side_by_side(im1,im2):
  f = plt.figure(figsize=(12,12),frameon=False)
  #f = plt.figure()
  f.add_subplot(1,2, 1)
  plt.imshow(np.rot90(im1,2))
  f.add_subplot(1,2, 2)
  plt.imshow(np.rot90(im2,2))
  plt.show(block=True)
  
  
  
def show_me_img(path):#show a single image
  tif=TIFF.open(path)
  image=tif.read_image()
  plt.imshow(image,interpolation='nearest')
  plt.show()


#This function shows evry image in the folder , sorted as image and mask
def show_me_folder(path):
  for file in sorted(os.listdir(path)):
    #if file.endswith(change):#on colab, deleting files will add some weird stuff on the folder
    base_direc=path+'/'
    print(file)
    tif=TIFF.open(base_direc+file)
    image=tif.read_image()
    plt.imshow(image,interpolation='nearest')
    plt.show()

#Visualisation of albumentations transform

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        
def save_to_tif(path, data):
    with open(path, 'wb') as f:
        np.save(f, data, allow_pickle=True) 
    


def load_set(folder, is_mask, shuffle=False):
    data = []
    img_list = sorted(glob.glob(os.path.join(folder, '*.tif')) +
                      glob.glob(os.path.join(folder, '*.jpg'))+
                      glob.glob(os.path.join(folder, '*.png'))+
                      glob.glob(os.path.join(folder, '*.tiff')))
    if shuffle:
        np.random.shuffle(img_list)
    for img_fn in img_list:
        img = load_image(img_fn, is_mask)
        data.append(img)
    return data, img_list



    
def load_image(path,is_mask):
    if not is_mask:
        return np.asarray(Image.open(path).convert("RGB"))
    else:
        return np.asarray(Image.open(path).convert('L'))


def load_set_txt(folder):
    data = []
    txt_list = sorted(glob.glob(os.path.join(folder, '*.txt')) +
                      glob.glob(os.path.join(folder, '*.csv')))
    
    for txt_fn in txt_list:

      txt = load_txt(txt_fn)
      data.append(txt)
    
    return data, txt_list

def load_txt(path):
  return np.loadtxt(path)


def create_dir(dirname):
    try:
        os.mkdir(dirname)
        return True
    except OSError:
        return False

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def resize_my_images(src, dst):

    #credits: https://evigio.com/post/resizing-images-into-squares-with-opencv-and-python
    

    i = 1
    img_size = 512
    path = src
    for img_name in sorted(os.listdir(path)):
        
        img = None
        print(img_name)

        img = cv2.imread(os.path.join(path, img_name),
                            cv2.IMREAD_GRAYSCALE)

        h, w = img.shape[:2]
        a1 = w/h
        a2 = h/w

        if(a1 > a2):
            
            w_target = round(img_size * a1)
            h_target = img_size

            r_img = cv2.resize(
                img, (w_target, h_target), interpolation=cv2.INTER_AREA)
            margin = int(r_img.shape[1] / 6)
            crop_img = r_img[0:img_size, margin:(margin+img_size)]

            

        elif(a1 < a2):
            
            w_target = img_size
            h_target = round(img_size * a2)

            r_img = cv2.resize(img, (w_target, h_target),
                              interpolation=cv2.INTER_AREA)
            margin = int(r_img.shape[0] / 6)
            crop_img = r_img[margin:(margin+img_size), 0:img_size]

            
        elif(a1 == a2):
            
            w_target = img_size
            h_target = img_size

            r_img = cv2.resize(img, (w_target, h_target),
                              interpolation=cv2.INTER_AREA)
            crop_img = r_img[0:img_size, 0:img_size]

            

        if(crop_img.shape[0] != img_size or crop_img.shape[1] != img_size):
            
            crop_img = r_img[0:img_size, 0:img_size]

        if(crop_img.shape[0] == img_size and crop_img.shape[1] == img_size):

          cv2.imwrite(dst + img_name, crop_img)

          i += 1
          
            

def tif_to_nparray(img):
  #tif=TIFF.open(img)
  #return tif.read_image()
    return  skimage.io.imread(img, plugin='tifffile')


def size(path):
  teut=Image.open(path)
  width, height = teut.size
  return width,height


def normalise_mask_set(mask, threshold):
  
  mask[mask > threshold] = 1
  mask[mask <= threshold] = 0
  return mask 



