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
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage

import cv2
import os
import parser as Parser

show_image = True

# define the augmentation
#seq1 = iaa.Sequential([ iaa.Affine(rotate=(-30,30))]) # Augmentation for images and masks
#seq2 = iaa.Sequential([ iaa.Dropout([0.1,0.5])])      # Augmentation for images

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
        
    def augment_train_data(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        for idx in range (len(self.train_imgs)):
            
            aug_img = np.zeros(shape=(512,512,3)) 
            aug_label = np.zeros(shape=(1))
        
            X = self.train_imgs[idx]
            Y = KeypointsOnImage([Keypoint(x=self.train_labels[idx][0]*512, y=self.train_labels[idx][1]*512),
                                     Keypoint(x=self.train_labels[idx][2]*512, y=self.train_labels[idx][3]*512),], shape=X.shape)

            seq = iaa.Sequential(
                [
            
                    # crop some of the images by 0-10% of their height/width
    #                sometimes(iaa.Crop(percent=(0, 0.1))),
                            #
                    # Execute 0 to 5 of the following (less important) augmenters per
                    # image. Don't execute all of them, as that would often be way too
                    # strong.
                    #
                    iaa.SomeOf((0, 5),
                        [
    #                        # Convert some images into their superpixel representation,
    #                        # sample between 20 and 200 superpixels per image, but do
    #                        # not replace all superpixels with their average, only
    #                        # some of them (p_replace).
    #                        sometimes(
    #                            iaa.Superpixels(
    #                                p_replace=(0, 1.0),
    #                                n_segments=(20, 200)
    #                            )
    #                        ),
            
                            # Blur each image with varying strength using
                            # gaussian blur (sigma between 0 and 3.0),
                            # average/uniform blur (kernel size between 2x2 and 7x7)
                            # median blur (kernel size between 3x3 and 11x11).
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 3.0)),
                                iaa.AverageBlur(k=(2, 7)),
#                                iaa.MedianBlur(k=(3, 11)),
                            ]),
            
                            # Sharpen each image, overlay the result with the original
                            # image using an alpha between 0 (no sharpening) and 1
                            # (full sharpening effect).
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            
                            # Same as sharpen, but for an embossing effect.
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            
                            # Search in some images either for all edges or for
                            # directed edges. These edges are then marked in a black
                            # and white image and overlayed with the original image
                            # using an alpha of 0 to 0.7.
    #                        sometimes(iaa.OneOf([
    #                            iaa.EdgeDetect(alpha=(0, 0.7)),
    #                            iaa.DirectedEdgeDetect(
    #                                alpha=(0, 0.7), direction=(0.0, 1.0)
    #                            ),
    #                        ])),
            
            
            
                            # Improve or worsen the contrast of images.
                            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            
                            # Convert each image to grayscale and then overlay the
                            # result with the original with random alpha. I.e. remove
                            # colors with varying strengths.
                            iaa.Grayscale(alpha=(0.0, 1.0)),
            
                            # In some images move pixels locally around (with random
                            # strengths).
                            sometimes(
                                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                            ),
            
                            # In some images distort local areas with varying strength.
                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                        ],
                        # do all of the above augmentations in random order
                        random_order=True
                    )
                ],
                # do all of the above augmentations in random order
                random_order=True
            )
            seq.deterministic = True
       
            aug_img, aug_label = seq(image=X,  keypoints=Y)
            
            aug_label = [(aug_label.keypoints[0].x/512), (aug_label.keypoints[0].y/512), 
                         (aug_label.keypoints[1].x/512), (aug_label.keypoints[1].y/512)]
     
            self.aug_train_imgs.append(aug_img)
            self.aug_train_labels.append(aug_label)
        
    def augment_test_data(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        for idx in range (len(self.test_imgs)):
            
            aug_img = np.zeros(shape=(512,512,3)) 
            aug_label = np.zeros(shape=(1))
        
            X = self.test_imgs[idx]
            Y = KeypointsOnImage([Keypoint(x=self.test_labels[idx][0]*512, y=self.test_labels[idx][1]*512),
                                      Keypoint(x=self.test_labels[idx][2]*512, y=self.test_labels[idx][3]*512)], shape=X.shape)

            seq = iaa.Sequential(
                [
            
                    # crop some of the images by 0-10% of their height/width
    #                sometimes(iaa.Crop(percent=(0, 0.1))),
                            #
                    # Execute 0 to 5 of the following (less important) augmenters per
                    # image. Don't execute all of them, as that would often be way too
                    # strong.
                    #
                    iaa.SomeOf((0, 5),
                        [
    #                        # Convert some images into their superpixel representation,
    #                        # sample between 20 and 200 superpixels per image, but do
    #                        # not replace all superpixels with their average, only
    #                        # some of them (p_replace).
    #                        sometimes(
    #                            iaa.Superpixels(
    #                                p_replace=(0, 1.0),
    #                                n_segments=(20, 200)
    #                            )
    #                        ),
            
                            # Blur each image with varying strength using
                            # gaussian blur (sigma between 0 and 3.0),
                            # average/uniform blur (kernel size between 2x2 and 7x7)
                            # median blur (kernel size between 3x3 and 11x11).
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 3.0)),
                                iaa.AverageBlur(k=(2, 7)),
#                                iaa.MedianBlur(k=(3, 11)),
                            ]),
            
                            # Sharpen each image, overlay the result with the original
                            # image using an alpha between 0 (no sharpening) and 1
                            # (full sharpening effect).
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            
                            # Same as sharpen, but for an embossing effect.
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            
                            # Search in some images either for all edges or for
                            # directed edges. These edges are then marked in a black
                            # and white image and overlayed with the original image
                            # using an alpha of 0 to 0.7.
                            sometimes(iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0, 0.7)),
                                iaa.DirectedEdgeDetect(
                                    alpha=(0, 0.7), direction=(0.0, 1.0)
                                ),
                            ])),        
            
                            # Improve or worsen the contrast of images.
                            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            
                            # Convert each image to grayscale and then overlay the
                            # result with the original with random alpha. I.e. remove
                            # colors with varying strengths.
                            iaa.Grayscale(alpha=(0.0, 1.0)),
            
                            # In some images move pixels locally around (with random
                            # strengths).
                            sometimes(
                                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                            ),
            
                            # In some images distort local areas with varying strength.
                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                        ],
                        # do all of the above augmentations in random order
                        random_order=True
                    )
                ],
                # do all of the above augmentations in random order
                random_order=True
            )              
            seq.deterministic = True
            
            aug_img, aug_label = seq(image=X,  keypoints=Y)
            
            aug_label = [(aug_label.keypoints[0].x/512), (aug_label.keypoints[0].y/512), 
                         (aug_label.keypoints[1].x/512), (aug_label.keypoints[1].y/512)]
            
            self.aug_test_imgs.append(aug_img)
            self.aug_test_labels.append(aug_label)
            
            
    def get_data_after_augment(self):

        
        self.augment_train_data()
        self.augment_test_data()
             
        ret_train_imgs = self.train_imgs
        ret_train_imgs = np.concatenate((ret_train_imgs, np.array(self.aug_train_imgs)), axis=0)
        
        ret_train_labels = self.train_labels
        ret_train_labels = np.concatenate((ret_train_labels, np.array(self.aug_train_labels)), axis=0)
        
        ret_test_imgs = self.test_imgs
        ret_test_imgs = np.concatenate((ret_test_imgs, np.array(self.aug_test_imgs)), axis=0)
        
        ret_test_labels = self.test_labels
#        print("ret_test_labels ",ret_test_labels.shape )
#        print("self.aug_test_labels ", np.array(self.aug_test_labels).shape)
#        print("self.aug_test_labels ", self.aug_test_labels)
        ret_test_labels = np.concatenate((ret_test_labels, np.array(self.aug_test_labels)), axis=0)
      
        
        return [ret_train_imgs, ret_train_labels, ret_test_imgs, ret_test_labels]
        
    
    def drawImageWithKeyPoints(self, idx):
        visualizeImageAndLabel(self.train_imgs[idx], self.train_labels[idx], 
                               self.aug_train_imgs[idx] , self.aug_train_labels[idx])

        
        
        


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
        
        #iaa.Multiply((1.2, 1.5))
        #,iaa.Affine(rotate=(-30,30)) 

        seq1 = iaa.Sequential([iaa.Affine(scale=(0.5, 0.7))])
        #seq2 = iaa.Sequential([iaa.Dropout(p=0.5)])                  
        seq1.deterministic = True
        
        image_augmented, label_augmented = seq1(image=X,  bounding_boxes=Y)
        #image_augmented[i] = seq2.augment_image(image_augmented[i])
  
        if show_image:
            visualizeImageAndLabel(self.train_imgs, self.test_labels, image_augmented ,label_augmented)

    
        
