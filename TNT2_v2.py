#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:54:02 2022

@author: yasinceran
"""

# -*- coding: utf-8 -*-
#!pip install tensorflow
# pip install opencv-python
#pip install segmentation_models



import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
import keras 

from keras.utils import normalize
from keras.metrics import MeanIoU

# check video 182 for glob

#!mkdir images images/{label,nolabel} 
#os.getcwd()
#os.listdir()
#os.chdir()
path="/Users/yasinceran/Library/CloudStorage/GoogleDrive-yaceran@gmail.com/My Drive/Research/glioblostama/TNT-2ndProject/"
os.chdir(path)
os.getcwd()
os.makedirs("images")
os.makedirs("images/label")
os.makedirs("images/nolabel")

######Copy the images to their destinations
source_folder = path+"msto-images/"
label_folder = path+"images/label/"
nolabel_folder = path+"images/nolabel/"

# fetch all files
for file_name in os.listdir(source_folder):
    # construct full file path
    source = source_folder + file_name
    if "label" in file_name:
        destination = label_folder + file_name
    else:
        destination = nolabel_folder+file_name
    # copy only files
    if os.path.isfile(source):
        shutil.copy(source, destination)
        print('copied', file_name)


#######form the smaller images for image preprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from skimage import io, filters
from scipy import ndimage
import sys
import imutils

os.makedirs("images/aura_img_mask")
os.makedirs("images/aura_img")
os.makedirs("images/img512")
os.makedirs("images/img512_mask")
os.makedirs("images/img256")

def draw_box(x,y,width,height,size,img,img_label,img_label_output,corner,filename,i,j):
    
    tnt = False
    
    imgbox = img[y:y+height,x:x+width,:]
    imgbox_label = img_label[y:y+height,x:x+width,:]
    
    imgbox_label_mid = img_label[y+corner:y+height-corner,x+corner:x+width-corner,:]
    
    ind_mid = np.where(imgbox_label_mid[:, :, 2]!=0) # checking if labels are in center
    ind = np.where(imgbox_label[:, :, 2]!=0) 

    #if len(ind_mid[0])>0 and size==512:
            #plt.imshow(imgbox_label)
            #plt.show()
            #print(len(ind_mid[0]))
    
    if size==512:
        if True: #len(ind_mid[0])>50:
            img_label_output = cv2.rectangle(img_label_output,(x, y),( x+width,y+height), (255,0,0),5)
            cv2.imwrite('./images/img512/'+filename+'_x'+str(x)+'_y'+str(y)+'_w'+str(width)+'_h'+str(height)+'_'+'1.tif', imgbox)
            cv2.imwrite('./images/img512_mask/'+filename+'_x'+str(x)+'_y'+str(y)+'_w'+str(width)+'_h'+str(height)+'.tif', imgbox_label)
            tnt = True
        elif (len(ind[0])<4) and (j==0 or j==7) and (i==0 or i==7):
            cv2.imwrite('./images/img512/'+filename+'_x'+str(x)+'_y'+str(y)+'_w'+str(width)+'_h'+str(height)+'_'+'0.tif', imgbox)
            
    elif size==256:
        if True: #len(ind_mid[0])>50:
            img_label_output = cv2.rectangle(img_label_output,(x, y),( x+width,y+height), (255,255,0),5)
            #cv2.imwrite('./images/img256/'+filename+'_x'+str(x)+'_y'+str(y)+'_w'+str(width)+'_h'+str(height)+'_'+'1.tif', imgbox)
            cv2.imwrite('./images/aura_img/'+filename+'_x'+str(x)+'_y'+str(y)+'_w'+str(width)+'_h'+str(height)+'.tif', imgbox)
            cv2.imwrite('./images/aura_img_mask/'+filename+'_x'+str(x)+'_y'+str(y)+'_w'+str(width)+'_h'+str(height)+'.tif', imgbox_label)
            tnt = True
        elif (len(ind[0])<1) and (j==0 or j==7) and (i==0 or i==7):
            cv2.imwrite('./images/img256/'+filename+'_x'+str(x)+'_y'+str(y)+'_w'+str(width)+'_h'+str(height)+'_'+'0.tif', imgbox)
            
                   
    return tnt, img_label_output


def prepare_boxes(splitw,splith,x,y,width,height,corner,size,img,img_label,img_label_output,loop,box_split,w,h,filename):
    loop -= 1
    num = 1 if loop==1 else 2
    for smw in range(0,splitw):
        for smh in range(0,splith):
            for i in range(0,num):
                for j in range(0,num):

                    xst = x + (width*smw) + (width//num*j)
                    yst = y + (height*smh) + (height//num*i)

                    if xst+width > w:
                        continue
                    if yst+height > h:
                        continue

                    tnt,img_label_out=draw_box(xst,yst,
                            width,height,
                            size,
                            img,img_label,img_label_output,corner,filename,i,j)
                    
                    if tnt and loop==1:
                        img_label_output = prepare_boxes(box_split,box_split,xst,yst,width//box_split,height//box_split,corner//box_split,size//box_split,img,img_label,img_label_output,loop,box_split,w,h,filename)
                    
    return img_label_output


def input_boxes(filename,rotated,path):
    img_label = cv2.imread(os.path.join(path+'label',filename))

    img = cv2.imread(os.path.join(path+'nolabel',filename[:-10]+'.png'))
    img = cv2.resize(img, (img_label.shape[1],img_label.shape[0]))

    img = cv2.copyMakeBorder(img,256,256,256,256,cv2.BORDER_REFLECT)

    img_label = cv2.copyMakeBorder(img_label,256,256,256,256,cv2.BORDER_REFLECT)

    img_label[np.where(img_label[:, :, 0]==img_label[:, :, 2])] = [0, 0, 0]
    img_label[np.where(img_label[:, :, 0]!=img_label[:, :, 2])] = [255, 255, 255]
    img_label = cv2.threshold(img_label, 128, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((9, 9), np.uint8)
    img_label = cv2.dilate(img_label, kernel) 
  
    img_label_output = cv2.imread(os.path.join(path+'label',filename))
    img_label_output = cv2.copyMakeBorder(img_label_output,256,256,256,256,cv2.BORDER_REFLECT)

    if (rotated):

        img = imutils.rotate(img, angle=10)[500:-500,400:-400,:]

        img_label = imutils.rotate(img_label, angle=10)[500:-500,400:-400,:]

        img_label_output = imutils.rotate(img_label_output, angle=10)[500:-500,400:-400,:]


    h,w,c = img.shape

    width = 512 #w//wsplit
    height = 512 #h//hsplit
    wsplit = w//width
    hsplit = h//height
    box_split=2
    
    
    corner=240
    box_size=512
    
    loop=2 #for different size image boxes
    
    return prepare_boxes(wsplit,hsplit,0,0,
                width,height,
                corner,box_size,
                img,img_label,img_label_output,loop,box_split,w,h,filename[:-10])


imagepath = path+"images/"
imagepaths = []
labelpaths = []

for p in sorted(os.listdir(imagepath+'nolabel/')):
  imagepaths.append(p)

for p in sorted(os.listdir(imagepath+'label/')):
  labelpaths.append(p)

print(imagepaths,labelpaths)


for index, filename in enumerate(labelpaths):
    print(filename)
    if(index<len(labelpaths)*0.5):
        img_label_output = input_boxes(filename,False,imagepath)
        
        fig =  plt.figure(figsize=(20,30),frameon=False)
        fig.tight_layout()
        plt.title(filename)
        plt.imshow(cv2.cvtColor(img_label_output, cv2.COLOR_BGR2RGB))
        #cv2.imwrite(path+'labelout/'+filename[:-10]+'out.png', img_label_output)
        plt.show()
        '''
        img_label_output = input_boxes(filename,True)
        
        fig =  plt.figure(figsize=(20,30),frameon=False)
        fig.tight_layout()
        plt.title(filename+' rotated')
        plt.imshow(cv2.cvtColor(img_label_output, cv2.COLOR_BGR2RGB))
        #cv2.imwrite(path+'labelout/'+filename[:-10]+'out.png', img_label_output)
        plt.show()
        '''

#Resizing images, if needed
SIZE_X = 512 
SIZE_Y = 512
n_classes=2 #Number of classes for segmentation

#Capture training image info as a list
train_images = []

imgpth=glob.glob(os.path.join(imagepath, "img512/*.tif"))


for img_path in imgpth:
    img = cv2.imread(img_path, 1)       
    #img = cv2.resize(img, (SIZE_Y, SIZE_X))
    train_images.append(img)
       
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 
maskpth=glob.glob(os.path.join(imagepath, "img512_mask/*.tif"))

for mask_path in maskpth:
    mask = cv2.imread(mask_path, 0)       
    #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
    train_masks.append(mask)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#################################################
#train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

#Further split training data t a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.5, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

######################################################
#Reused parameters in all models

n_classes=2
activation='softmax'

LR = 0.0001
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


########################################################################
###Model 1
BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)

# define model
model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model1.compile(optim, total_loss, metrics=metrics)

#model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model1.summary())


history1=model1.fit(X_train1, 
          y_train_cat,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test1, y_test_cat))


model1.save('res34_backbone_50epochs_v2_1.hdf5')
############################################################
###Model 2

BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

# preprocess input
X_train2 = preprocess_input2(X_train)
X_test2 = preprocess_input2(X_test)

# define model
model2 = sm.Unet(BACKBONE2, encoder_weights='imagenet', classes=n_classes, activation=activation)


# compile keras model with defined optimozer, loss and metrics
model2.compile(optim, total_loss, metrics)
#model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)


print(model2.summary())


history2=model2.fit(X_train2, 
          y_train_cat,
          batch_size=8, 
          epochs=10,
          verbose=1,
          validation_data=(X_test2, y_test_cat))

#tf.saved_model.save(resnet50v2, 'inceptionv3_backbone_120epochs_v2_1.hdf5')  
model2.save('inceptionv3_backbone_10epochs_v2_1.hdf5')

#####################################################
###Model 3

BACKBONE3 = 'vgg16'
preprocess_input3 = sm.get_preprocessing(BACKBONE3)

# preprocess input
X_train3 = preprocess_input3(X_train)
X_test3 = preprocess_input3(X_test)


# define model
model3 = sm.Unet(BACKBONE3, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model3.compile(optim, total_loss, metrics)
#model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)


print(model3.summary())

history3=model3.fit(X_train3, 
          y_train_cat,
          batch_size=8, 
          epochs=10,
          verbose=1,
          validation_data=(X_test3, y_test_cat))


model3.save('vgg19_backbone_10epochs_v2_1.hdf5')


##########################################################

###
#plot the training and validation accuracy and loss at each epoch
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history1.history['iou_score']
val_acc = history1.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

#####################################################

from keras.models import load_model

### FOR NOW LET US FOCUS ON A SINGLE MODEL

#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('saved_models/res34_backbone_50epochs_v2_1.hdf5', compile=False)
model2 = load_model('saved_models/inceptionv3_backbone_10epochs_v2_1.hdf5', compile=False)
model3 = load_model('saved_models/vgg19_backbone_10epochs_v2_1.hdf5', compile=False)

#Weighted average ensemble
models = [model1, model2, model3]
#preds = [model.predict(X_test) for model in models]

pred1 = model1.predict(X_test1)
pred2 = model2.predict(X_test2)
pred3 = model3.predict(X_test3)

preds=np.array([pred1, pred2, pred3])

#preds=np.array(preds)
weights = [0.3, 0.5, 0.2]

#Use tensordot to sum the products of all elements over specified axes.
weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=3)

y_pred1_argmax=np.argmax(pred1, axis=3)
y_pred2_argmax=np.argmax(pred2, axis=3)
y_pred3_argmax=np.argmax(pred3, axis=3)


#Using built in keras function
n_classes = 2
IOU1 = MeanIoU(num_classes=n_classes)  
IOU2 = MeanIoU(num_classes=n_classes)  
IOU3 = MeanIoU(num_classes=n_classes)  
IOU_weighted = MeanIoU(num_classes=n_classes)  

IOU1.update_state(y_test[:,:,:,0], y_pred1_argmax)
IOU2.update_state(y_test[:,:,:,0], y_pred2_argmax)
IOU3.update_state(y_test[:,:,:,0], y_pred3_argmax)
IOU_weighted.update_state(y_test[:,:,:,0], weighted_ensemble_prediction)


print('IOU Score for model1 = ', IOU1.result().numpy())
print('IOU Score for model2 = ', IOU2.result().numpy())
print('IOU Score for model3 = ', IOU3.result().numpy())
print('IOU Score for weighted average ensemble = ', IOU_weighted.result().numpy())
###########################################
#Grid search for the best combination of w1, w2, w3 that gives maximum acuracy

import pandas as pd
df = pd.DataFrame([])

for w1 in range(0, 4):
    for w2 in range(0,4):
        for w3 in range(0,4):
            wts = [w1/10.,w2/10.,w3/10.]
            
            IOU_wted = MeanIoU(num_classes=n_classes) 
            wted_preds = np.tensordot(preds, wts, axes=((0),(0)))
            wted_ensemble_pred = np.argmax(wted_preds, axis=3)
            IOU_wted.update_state(y_test[:,:,:,0], wted_ensemble_pred)
            print("Now predciting for weights :", w1/10., w2/10., w3/10., " : IOU = ", IOU_wted.result().numpy())
            df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
                                         'wt3':wts[2], 'IOU': IOU_wted.result().numpy()}, index=[0]), ignore_index=True)
            
max_iou_row = df.iloc[df['IOU'].idxmax()]
print("Max IOU of ", max_iou_row[3], " obained with w1=", max_iou_row[0],
      " w2=", max_iou_row[1], " and w3=", max_iou_row[2])         


#############################################################
opt_weights = [max_iou_row[0], max_iou_row[1], max_iou_row[2]]

#Use tensordot to sum the products of all elements over specified axes.
opt_weighted_preds = np.tensordot(preds, opt_weights, axes=((0),(0)))
opt_weighted_ensemble_prediction = np.argmax(opt_weighted_preds, axis=3)
#######################################################
#Predict on a few images

import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,:]
test_img_input=np.expand_dims(test_img_norm, 0)

#Weighted average ensemble
models = [model1, model2, model3]

test_img_input1 = preprocess_input1(test_img_input)
test_img_input2 = preprocess_input2(test_img_input)
test_img_input3 = preprocess_input3(test_img_input)

test_pred1 = model1.predict(test_img_input1)
test_pred2 = model2.predict(test_img_input2)
test_pred3 = model3.predict(test_img_input3)

test_preds=np.array([test_pred1, test_pred2, test_pred3])

#Use tensordot to sum the products of all elements over specified axes.
weighted_test_preds = np.tensordot(test_preds, opt_weights, axes=((0),(0)))
weighted_ensemble_test_prediction = np.argmax(weighted_test_preds, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(weighted_ensemble_test_prediction, cmap='jet')
plt.show()

#####################################################################

import utilities


label_image = cv2.imread('images/label/m02-label.png')
original_image = cv2.imread('images/nolabel/m02.png')
imgsize=512

img_label = cv2.copyMakeBorder(label_image,imgsize,imgsize,imgsize,imgsize,cv2.BORDER_REFLECT)

img_label[np.where(img_label[:, :, 0]==img_label[:, :, 2])] = [0, 0, 0]
img_label[np.where(img_label[:, :, 0]!=img_label[:, :, 2])] = [255, 255, 255]
img_label = cv2.threshold(img_label, 128, 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((9, 9), np.uint8)
img_label = cv2.dilate(img_label, kernel)
                       
img_orig = cv2.copyMakeBorder(original_image,imgsize,imgsize,imgsize,imgsize,cv2.BORDER_REFLECT)



import numpy as np
from skimage.util import view_as_blocks

img_label.shape[0]-img_label.shape[0]//imgsize*imgsize

masks_blocks = view_as_blocks(img_label[79:,139:,:],(imgsize,imgsize,3)).reshape(-1,imgsize,imgsize,3)
image_blocks = view_as_blocks(img_orig[79:,139:,:],(imgsize,imgsize,3)).reshape(-1,imgsize,imgsize,3)



masks_blocks = view_as_blocks(img_label[:5120,:6656,:],(imgsize,imgsize,3)).reshape(-1,imgsize,imgsize,3)
image_blocks = view_as_blocks(img_orig[:5120,:6656,:],(imgsize,imgsize,3)).reshape(-1,imgsize,imgsize,3)





#to plot all images in the validation set, and you can do the same with the test set using dataloader_test['test'].
for epoch in range(260):
        truth=masks_blocks[epoch]
        img=image_blocks[epoch]
        if np.sum(truth) == 0:
          continue

        
        normalize_img = transforms.Compose([transforms.ToTensor()])
        image = normalize_img(img).unsqueeze(0)

        img = image.to(device)
        proba_prediction= trainer.predict(img)
        print('###########################################################')
        print('###########################################################')
        print('###########################################################')
        tmp=proba_prediction.squeeze()
        normalizedImg = np.zeros((256, 256))#put the size of your images here
        normalizedImg = cv2.normalize(tmp,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
        mask_pred=normalise_mask_set(normalizedImg,0.75)
        
        plot_side_by_side(to_numpy(img.permute(0,3,2,1).squeeze()),proba_prediction.squeeze())
        plot_side_by_side(np.transpose(truth, (1,0,2)),mask_pred.squeeze())
       
        print('###########################################################')
        print('###########################################################')
        print('###########################################################')









