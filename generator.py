# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:49:35 2020

@author: rglabor07
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import numpy as np
from os import listdir

from keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage

import cv2
import os
import parser as Parser

show_image = True

def visualizeImageAndLabel(image, label, image_aug, label_aug):
#    image_before = label.draw_on_image(image, size = 2)
#    image_after = label_aug.draw_on_image(image_aug, size=2, color=[255, 255, 255])
    fig,ax = plt.subplots(1)
    image = Image.fromarray(image, 'RGB')
#    plt.title("image")
    ax.imshow(image)
    
    yValue = label[0]*512
    xValue = label[1]*512
    width = label[2]*512 - yValue
    height = label[3]*512 - xValue
    rect = patches.Rectangle((xValue,yValue),width,height,linewidth=1,edgecolor='r',facecolor='none')

    ax.add_patch(rect)
    
    fig2,ax2 = plt.subplots(1)
#    plt.figure(2001)
#    plt.title("image augmented")
    image_aug = Image.fromarray(image_aug, 'RGB')
    ax2.imshow(image_aug)
    
    yValueLabel = label_aug[0]*512
    xValueLabel = label_aug[1]*512
    widthLabel = label_aug[2]*512 - yValueLabel
    heightLabel = label_aug[3]*512 - xValueLabel
    
    rectLabel = patches.Rectangle((xValueLabel,yValueLabel),widthLabel,heightLabel,linewidth=1,edgecolor='b',facecolor='none')

    ax2.add_patch(rectLabel)
    plt.show()
    
    
    
#aufruf der klasse mit x_path, y_label und der batchsize
#unterscheidung zwischen 2d und 3d?? mit einem flag oder wie?
class Generator():
        
    def __init__(self, train_imgs, train_labels, test_imgs, test_labels):
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.test_imgs = test_imgs
        self.test_labels = test_labels
        
        self.aug_train_imgs = []
        self.aug_train_labels = []
        self.aug_test_imgs = []
        self.aug_test_labels = []        
        
    def augment_2D_train_data(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        for loop in range(2):
          print("augment train data loop: ", loop)
          for idx in range (len(self.train_imgs)):
              
              aug_img = np.zeros(shape=(512,512,3)) 
              aug_label = np.zeros(shape=(1))
          
              X = self.train_imgs[idx]
              Y = KeypointsOnImage([Keypoint(x=self.train_labels[idx][0]*512, y=self.train_labels[idx][1]*512),
                                      Keypoint(x=self.train_labels[idx][2]*512, y=self.train_labels[idx][3]*512),], shape=X.shape)

              seq2D = iaa.Sequential([
                  # Small gaussian blur with random sigma between 0 and 0.5.
                  # But we only blur about 50% of all images.
                  iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                  ),
                  # Strengthen or weaken the contrast in each image.
                  iaa.ContrastNormalization((0.75, 1.5)),
                  # Add gaussian noise.
                  # For 50% of all images, we sample the noise once per pixel.
                  # For the other 50% of all images, we sample the noise per pixel AND
                  # channel. This can change the color (not only brightness) of the
                  # pixels.
                  iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                  # Make some images brighter and some darker.
                  # In 20% of all cases, we sample the multiplier once per channel,
                  # which can end up changing the color of the images.
                  iaa.Multiply((0.8, 1.2), per_channel=0.2),
                  # Apply affine transformations to each image.
                  # Scale/zoom them, translate/move them, rotate them and shear them.
                  # iaa.Affine(
                  #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                  #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                  # )
              ], random_order=True) # apply augmenters in random order

              seq2D.deterministic = True            

              aug_img, aug_label = seq2D(image=X,  keypoints=Y)

              aug_label = [(aug_label.keypoints[0].x/512), (aug_label.keypoints[0].y/512), 
                          (aug_label.keypoints[1].x/512), (aug_label.keypoints[1].y/512)]
      
              self.aug_train_imgs.append(aug_img)
              self.aug_train_labels.append(aug_label)
        
    def augment_2D_test_data(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        for loop in range(2):
          print("augment test data loop: ", loop)
          for idx in range (len(self.test_imgs)):
              
              aug_img = np.zeros(shape=(512,512,3)) 
              aug_label = np.zeros(shape=(1))
          
              X = self.test_imgs[idx]
              Y = KeypointsOnImage([Keypoint(x=self.train_labels[idx][0]*512, y=self.train_labels[idx][1]*512),
                                      Keypoint(x=self.train_labels[idx][2]*512, y=self.train_labels[idx][3]*512),], shape=X.shape)
              seq = iaa.Sequential([
                  # Small gaussian blur with random sigma between 0 and 0.5.
                  # But we only blur about 50% of all images.
                  iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                  ),
                  # Strengthen or weaken the contrast in each image.
                  iaa.ContrastNormalization((0.75, 1.5)),
                  # Add gaussian noise.
                  # For 50% of all images, we sample the noise once per pixel.
                  # For the other 50% of all images, we sample the noise per pixel AND
                  # channel. This can change the color (not only brightness) of the
                  # pixels.
                  iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                  # Make some images brighter and some darker.
                  # In 20% of all cases, we sample the multiplier once per channel,
                  # which can end up changing the color of the images.
                  iaa.Multiply((0.8, 1.2), per_channel=0.2),
                  # Apply affine transformations to each image.
                  # Scale/zoom them, translate/move them, rotate them and shear them.
                  # iaa.Affine(
                  #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                  #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},

                  # )
              ], random_order=True) # apply augmenters in random order
              
              aug_img, aug_label = seq(image=X,  keypoints=Y)
              
              aug_label = [(aug_label.keypoints[0].x/512), (aug_label.keypoints[0].y/512), 
                          (aug_label.keypoints[1].x/512), (aug_label.keypoints[1].y/512)]
              
              self.aug_test_imgs.append(aug_img)
              self.aug_test_labels.append(aug_label)

    def augment_3D_train_data(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        for idx in range (len(self.train_imgs)):
            
            aug_img = np.zeros(shape=(512,512,3)) 
            aug_label = np.zeros(shape=(1))
        
            X = self.train_imgs[idx]
            Y = KeypointsOnImage([Keypoint(x=self.train_labels[idx][0]*512, y=self.train_labels[idx][1]*512),
                                      Keypoint(x=self.train_labels[idx][2]*512, y=self.train_labels[idx][3]*512),
                                      Keypoint(x=self.train_labels[idx][4]*512, y=self.train_labels[idx][5]*512),
                                      Keypoint(x=self.train_labels[idx][6]*512, y=self.train_labels[idx][7]*512),
                                      Keypoint(x=self.train_labels[idx][8]*512, y=self.train_labels[idx][9]*512),
                                      Keypoint(x=self.train_labels[idx][10]*512, y=self.train_labels[idx][11]*512),
                                      Keypoint(x=self.train_labels[idx][12]*512, y=self.train_labels[idx][13]*512),
                                      Keypoint(x=self.train_labels[idx][14]*512, y=self.train_labels[idx][15]*512)],
                                      shape=X.shape)

            seq2D = iaa.Sequential([
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(0.5,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                # Strengthen or weaken the contrast in each image.
                iaa.ContrastNormalization((0.75, 1.5)),
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                )
            ], random_order=True) # apply augmenters in random order

            seq2D.deterministic = True            

            aug_img, aug_label = seq2D(image=X,  keypoints=Y)    

            aug_label = [(aug_label.keypoints[0].x/512), (aug_label.keypoints[0].y/512), 
                         (aug_label.keypoints[1].x/512), (aug_label.keypoints[1].y/512),
                         (aug_label.keypoints[2].x/512), (aug_label.keypoints[2].y/512),
                         (aug_label.keypoints[3].x/512), (aug_label.keypoints[3].y/512),
                         (aug_label.keypoints[4].x/512), (aug_label.keypoints[4].y/512),
                         (aug_label.keypoints[5].x/512), (aug_label.keypoints[5].y/512),
                         (aug_label.keypoints[6].x/512), (aug_label.keypoints[6].y/512),
                         (aug_label.keypoints[7].x/512), (aug_label.keypoints[7].y/512)]
     
            self.aug_train_imgs.append(aug_img)
            self.aug_train_labels.append(aug_label)
        
    def augment_3D_test_data(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        for idx in range (len(self.test_imgs)):
            
            aug_img = np.zeros(shape=(512,512,3)) 
            aug_label = np.zeros(shape=(1))
        
            X = self.test_imgs[idx]
            Y = KeypointsOnImage([Keypoint(x=self.test_labels[idx][0]*512, y=self.test_labels[idx][1]*512),
                                      Keypoint(x=self.test_labels[idx][2]*512, y=self.test_labels[idx][3]*512),
                                      Keypoint(x=self.test_labels[idx][4]*512, y=self.test_labels[idx][5]*512),
                                      Keypoint(x=self.test_labels[idx][6]*512, y=self.test_labels[idx][7]*512),
                                      Keypoint(x=self.test_labels[idx][8]*512, y=self.test_labels[idx][9]*512),
                                      Keypoint(x=self.test_labels[idx][10]*512, y=self.test_labels[idx][11]*512),
                                      Keypoint(x=self.test_labels[idx][12]*512, y=self.test_labels[idx][13]*512),
                                      Keypoint(x=self.test_labels[idx][14]*512, y=self.test_labels[idx][15]*512)],
                                      shape=X.shape)
            seq = iaa.Sequential([
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(0.5,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                # Strengthen or weaken the contrast in each image.
                iaa.ContrastNormalization((0.75, 1.5)),
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},

                )
            ], random_order=True) # apply augmenters in random order
            
            aug_img, aug_label = seq(image=X,  keypoints=Y)
            
            aug_label = [(aug_label.keypoints[0].x/512), (aug_label.keypoints[0].y/512), 
                         (aug_label.keypoints[1].x/512), (aug_label.keypoints[1].y/512),
                         (aug_label.keypoints[2].x/512), (aug_label.keypoints[2].y/512),
                         (aug_label.keypoints[3].x/512), (aug_label.keypoints[3].y/512),
                         (aug_label.keypoints[4].x/512), (aug_label.keypoints[4].y/512),
                         (aug_label.keypoints[5].x/512), (aug_label.keypoints[5].y/512),
                         (aug_label.keypoints[6].x/512), (aug_label.keypoints[6].y/512),
                         (aug_label.keypoints[7].x/512), (aug_label.keypoints[7].y/512)]
     
            
            self.aug_test_imgs.append(aug_img)
            self.aug_test_labels.append(aug_label)
            
            
    def get_data_after_augment(self, dimensions):
        
        if dimensions == '2D':
            self.augment_2D_train_data()
            self.augment_2D_test_data()
          
        elif dimensions == '3D':
            self.augment_3D_train_data()
            self.augment_3D_test_data()
            
        else: 
            return
             
        ret_train_imgs = self.train_imgs
        ret_train_imgs = np.concatenate((ret_train_imgs, np.array(self.aug_train_imgs)), axis=0)
        
        ret_train_labels = self.train_labels
        ret_train_labels = np.concatenate((ret_train_labels, np.array(self.aug_train_labels)), axis=0)
        
        ret_test_imgs = self.test_imgs
        ret_test_imgs = np.concatenate((ret_test_imgs, np.array(self.aug_test_imgs)), axis=0)
        
        ret_test_labels = self.test_labels
        ret_test_labels = np.concatenate((ret_test_labels, np.array(self.aug_test_labels)), axis=0)
      
        
        return [ret_train_imgs, ret_train_labels, ret_test_imgs, ret_test_labels]
        
    
    def drawImageWithKeyPoints(self, idx):
        visualizeImageAndLabel(self.train_imgs[idx], self.train_labels[idx], 
                               self.aug_train_imgs[idx] , self.aug_train_labels[idx])

    
        
