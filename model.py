# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:23:48 2019

@author: rglabor07
"""
root_path = 'drive/My Drive/Colab_Notebooks'  #change dir to your project folder

import tensorflow as tf

import numpy as np
from keras.layers import BatchNormalization, Activation, Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, UpSampling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from os import listdir
import os.path
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from jsonParser import Parser
#from root_path + "generator.ipynb" import Generator

# how to extract data
#import zipfile
#zip_ref = zipfile.ZipFile(root_path + "/synthetic.zip", 'r')
#zip_ref.extractall("/tmp")
#zip_ref.close()

# train data and labels
directory_path = './syntheticTest/'
data = listdir(directory_path)
#print(data)
x_paths = []
y_paths = []
data = np.array(data)
for dataIdx in range (len(data)):
    if os.path.splitext(data[dataIdx])[1][1:] == "png":
        x_paths.append(directory_path + data[dataIdx])
    elif os.path.splitext(data[dataIdx])[1][1:] == "json":
        y_paths.append(directory_path + data[dataIdx])
x_paths = np.array(x_paths)
y_paths = np.array(y_paths)

x_img = np.zeros(shape=(len(x_paths),512,512,3))
for idx in range(0, len(x_paths)):
  #print("idx= ", idx)
  tmp = cv2.resize(cv2.imread(x_paths[idx]),(512,512))
  x_img[idx] = tmp
print("durch mit x_img schleife")
#print(x_img[0])

x_img = np.array(x_img)
#print("x_img: ", x_img)

y_2D_labelBox = []
y_3D_labelBox = []
y_centerPoint_Label = []


for y_path in y_paths:
    #print(y_path)
    parser = Parser(y_path)
    y_2D_labelBox.append(parser.get_2D_data())
    y_3D_labelBox.append(parser.get_3D_data())
    y_centerPoint_Label.append(parser.get_centerPoint())

y_2D_labelBox = np.array(y_2D_labelBox)
y_3D_labelBox = np.array(y_3D_labelBox)

print("parser durch")
#print("2D label: ",y_2D_labelBox)
#print("3D label: ", y_3D_labelBox)

def getCreated2DModel():
  inputs = Input((512,512,3))   #evtl anpassen

  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
  pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
  
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
  pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
  
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(64, (1, 1), activation='relu', padding='same')(conv3)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)

  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(128, (1, 1), activation='relu', padding='same')(conv4)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)

  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
  conv5 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv5)
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
  conv5 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv5)
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
  pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv5)

  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
  conv6 = Conv2D(512, (1, 1), activation='relu', padding='same')(conv6)
  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)
  conv6 = Conv2D(512, (1, 1), activation='relu', padding='same')(conv6)
  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)

  conv7 = Conv2D(1000, (1, 1), activation='relu', padding='same')(conv6)
  avgpool = GlobalAveragePooling2D()(conv7)

  outputs = Dense(4, activation='softmax')(avgpool)

  model = Model(inputs=inputs, outputs=outputs)

  return model


def getCreated3DModel():
  inputs = Input((512,512,3))   #evtl anpassen

  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
  pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
  pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(64, (1, 1), activation='relu', padding='same')(conv3)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)

  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(128, (1, 1), activation='relu', padding='same')(conv4)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)


  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
  conv5 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv5)
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
  conv5 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv5)
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
  pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv5)

  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
  conv6 = Conv2D(512, (1, 1), activation='relu', padding='same')(conv6)
  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)
  conv6 = Conv2D(512, (1, 1), activation='relu', padding='same')(conv6)
  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)
  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)
  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)

  conv7 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)
  avgpool = GlobalAveragePooling2D()(conv7)
  # output muss mehr als 4 sein da wir für die 3d Boundingbox 24 werte brauchen
  outputs = Dense(24, activation='softmax')(avgpool)

  model = Model(inputs=inputs, outputs=outputs)

  return model

model = getCreated2DModel()

print("vor compile")
model.compile(optimizer='adam', loss='binary_crossentropy')
print("nach compile")
#generator = Generator(x_paths, y_2D_labelBox, 16) # yPaths brauchen wir nicht
#histGenerator = model.fit_generator(generator, steps_per_epoch=1000, epochs=10)
print("vor fit")
hist = model.fit(x_img, y_2D_labelBox, batch_size=8, epochs=3)
print("nach fit")

plt.figure(200)
plt.title("Loss")
plt.plot(hist.history["loss"])

plt.figure(201)
plt.title("val_loss")
plt.plot(hist.history["val_loss"])



#x_neuBilder
#for bilder in x_neuBilder:
#  (x,y, breite, hoehe) = hist.predict(x) 
#  cv2.drawRactangle(x,y,breite,hoehe)


#(x, y, breite, hoehe) = hist.predict(bild1)
#cv2.drawRactangle(x, y, breite,hoehe)