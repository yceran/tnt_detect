# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:58:56 2020

@author: ethan
"""
import os 
import numpy as np
from PIL import Image
#from libtiff import TIFF
import skimage.io
import glob
import cv2
import matplotlib.pyplot as plt


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












