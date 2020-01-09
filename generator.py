# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:49:35 2020

@author: rglabor07
"""

import tensorflow as tf

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

import cv2


show_image = False

# define the augmentation

def visualizeImageAndLabel(image, label):
    fig,ax = plt.subplots(1)
    plt.title("image")
    ax.imshow(image)

    yValueLabel = label[0]
    xValueLabel = label[1]
    widthLabel = label[2]
    heightLabel = label[3]
    rectLabel = patches.Rectangle((xValueLabel, yValueLabel), widthLabel, heightLabel, linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rectLabel)
    
    plt.show()

#aufruf der klasse mit x_path, y_path und der batchsize
#unterscheidung zwischen 2d und 3d?? mit einem flag oder wie?
class Generator(Sequence):

    def __init__(self, x_paths, y_paths, batchsize, scale=1/255, dim=(32,32,32), validation = False,n_channels=1, n_classes=10, shuffle=True):
        self.x_path = x_paths 
        self.y_path = y_paths
        self.scale = scale
        self.batchsize = batchsize
        # aenderbare default values
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.validation = validation

    def __len__(self):
        return int(np.ceil(len(self.x_path) / float(self.batchsize))) #

    def __getitem__(self, idx):
#        indexes = self.indexes[idx*self.batchsize:(idx+1)*self.batchsize]     
#        list_IDS_temp = [self.x_path[k] for k in indexes]
#        X,y = self.__data_generation(list_IDS_temp)
        
        X = np.zeros(shape=(self.batchsize, 512,512,3))
        
        # hier muss zwischen 2d und 3d unterschieden werden da
        # die anzahl der werte im Array unterschielich lang ist und somit 
        # muss die array anderst aufgebaut werden um anschließend verändert werden zu können
        Y = np.zeros(shape=(self.batchsize, 512,512)) 
#        print("shape of X: ", X.shape)
#        print("shape of Y: ", Y.shape)
        image_augmented = np.zeros(shape=(self.batchsize, 512,512,3)) 
        label_augmented = np.zeros(shape=(self.batchsize, 512,512))
        #TODOO ab hier anpassen 
        try:
            for i in range (0, self.batchsize):
                current_idx = (idx * self.batchsize + i)
                X[i] = cv2.resize(cv2.imread(self.x_path[current_idx]),(512,512))
                
                Y[i] = cv2.resize(cv2.imread(self.y_path[current_idx])[:,:,0],(512,512))#[:,:,:,0]
                
                if not self.validation:                   
                    seq1 = iaa.Sequential([ iaa.Affine(rotate=(-30,30))])
                    #seq2 = iaa.Sequential([iaa.Dropout(p=0.5)])            
                            
                    seq1.deterministic = True
                    
                    image_augmented[i] = seq1.augment_image(image=X[i])
                    #image_augmented[i] = seq2.augment_image(image_augmented[i])
                    label_augmented[i] = seq1.augment_image(image=Y[i])
                    
    #                print("img.shape = ",image_augmented[i].shape)
    #                print("mask.shape = ",mask_augmented[i].shape)
    
                    #visualizeImageAndMask(image_augmented,mask_augmented)
    
    
            if show_image:
#                bild1 = Y[0]  
#                plt.imshow(X[0])
#                plt.imshow(Y[0])
                visualizeImageAndLabel(image_augmented ,label_augmented)
                
            X = np.concatenate((X, image_augmented), axis=0)
            Y = np.concatenate((Y, label_augmented), axis=0)
            # von y die 2 dimensionen weg lasssen = 512,512,1
#            print(Y.shape)
#            Y = Y[:,:,:,0]
#            print(Y.shape)
#            np.reshape(Y, (64,512, 512, 1))
        except IndexError:
            print("index error", current_idx)
            pass
            
            
        return X,Y[...,np.newaxis]
