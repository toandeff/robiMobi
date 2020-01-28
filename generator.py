# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:49:35 2020

@author: rglabor07
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from os import listdir

from keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import cv2
import os
import parser as Parser

show_image = True

# define the augmentation
#seq1 = iaa.Sequential([ iaa.Affine(rotate=(-30,30))]) # Augmentation for images and masks
seq2 = iaa.Sequential([ iaa.Dropout([0.1,0.5])])      # Augmentation for images

def visualizeImageAndLabel(image, label, image_aug, label_aug):
    image_before = label.draw_on_image(image, size = 2)
    image_after = label_aug.draw_on_image(image_aug, size=2, color=[255, 255, 255])
 
    
    plt.figure(2000)
    plt.title("image")
    plt.imshow(image_before)
    
    plt.figure(2001)
    plt.title("image augmented")
    plt.imshow(image_after)
    
#aufruf der klasse mit x_path, y_label und der batchsize
#unterscheidung zwischen 2d und 3d?? mit einem flag oder wie?
class Generator():
        
    def __init__(self, imgs, labels):
        self.images = imgs
        self.labels = labels
        for idx in range(len(self.images)):
            self.__getitem__(idx)


    def __getitem__(self, idx):
        

        image_augmented = np.zeros(shape=(512,512,3)) 
        label_augmented = np.zeros(shape=(1))

        X = self.images[idx]
        
        Y = BoundingBoxesOnImage([
            BoundingBox(x1=self.labels[idx][0], y1=self.labels[idx][1], 
                        x2=self.labels[idx][2], y2=self.labels[idx][3]),], shape=X[idx].shape)
        self.labels[idx]
        print("labels values: (%.4f, %.4f, %.4f, %.4f)"%( 
              self.labels[idx][0], self.labels[idx][1], self.labels[idx][2], self.labels[idx][3]))

        seq1 = iaa.Sequential([iaa.Multiply((1.2, 1.5)) # change brightness
                               ,iaa.Affine(translate_px={"x":40, "y":60} , scale=(0.5, 0.7))             
#                                  ,iaa.Affine(rotate=(-30,30)) 
                               ])
        #seq2 = iaa.Sequential([iaa.Dropout(p=0.5)])                  
        seq1.deterministic = True
        
        image_augmented, label_augmented = seq1(image=X,  bounding_boxes=Y)
        #image_augmented[i] = seq2.augment_image(image_augmented[i])
  
        if show_image:
            visualizeImageAndLabel(X, Y, image_augmented ,label_augmented)

    
        
