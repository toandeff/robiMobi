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
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from os import listdir
import os.path
import cv2
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from jsonParser import Parser
from generator import Generator
#from root_path + "generator.ipynb" import Generator
# how to extract data
#import zipfile
#zip_ref = zipfile.ZipFile(root_path + "/synthetic.zip", 'r')
#zip_ref.extractall("/tmp")
#zip_ref.close()
generateData = True
# train data and labels
directory_path = './synthetic/'
data = listdir(directory_path)
#print(data)
x_paths = []
y_paths = []
data = np.array(data)
print("schleife beginnt jetzt")
for dataIdx in range (len(data)):
    if os.path.splitext(data[dataIdx])[1][1:] == "png":
        x_paths.append(directory_path + data[dataIdx])
    elif os.path.splitext(data[dataIdx])[1][1:] == "json":
        y_paths.append(directory_path + data[dataIdx])
x_paths = np.array(x_paths)
y_paths = np.array(y_paths)

splitTrainValData = int(len(x_paths) * 0.7) # splits the data to get 70 % Training Data and 30 % Validation Data

if len(x_paths) * 0.7 < 1:
    splitTrainValData = 1

x_pathsTrainData = x_paths[0:splitTrainValData]
x_pathsValData = x_paths[splitTrainValData:]
print(len(x_pathsTrainData), " " ,len(x_pathsValData))
x_imgTrainData = np.zeros(shape=(len(x_pathsTrainData),512,512,3), dtype=np.uint8)
x_imgValData = np.zeros(shape=(len(x_pathsValData),512,512,3), dtype=np.uint8)


for idx in range(0, len(x_pathsTrainData)):
  #print("idx= ", idx)
  tmp = cv2.resize(cv2.imread(x_paths[idx]),(512,512))
  x_imgTrainData[idx] = tmp

for idx in range(0, len(x_pathsValData)):
  tmp = cv2.resize(cv2.imread(x_paths[idx]),(512,512))
  x_imgValData[idx] = tmp
print("durch mit x_img schleife")
#print(x_img[0])

x_imgTrainData = np.array(x_imgTrainData)
x_imgValData = np.array(x_imgValData)


y_2D_labelBox = []
y_3D_labelBox = []
y_centerPoint_Label = []


    
for y_path in y_paths:
    #print(y_path)
    parser = Parser(y_path)
    y_2D_labelBox.append(parser.get_2D_data())
    y_3D_labelBox.append(parser.get_3D_data())
    y_centerPoint_Label.append(parser.get_centerPoint())
    

# split data into training data and test data

y_2D_labelBox = np.array(y_2D_labelBox)
y_3D_labelBox = np.array(y_3D_labelBox)

y_2D_labelTrainData =  y_2D_labelBox[0:splitTrainValData]
y_2D_labelValData = y_2D_labelBox[splitTrainValData:]

y_3D_labelTrainData = y_3D_labelBox[0:splitTrainValData]
y_3D_labelValData =  y_3D_labelBox[splitTrainValData:]


if generateData:
    gen_2d = Generator(x_imgTrainData, y_2D_labelTrainData, x_imgValData, y_2D_labelValData) # return (x_img append new_img)
    [x_imgTrainData, y_2D_labelTrainData, x_imgValData, y_2D_labelValData] = gen_2d.get_data_after_augment()

print("parser durch")
def getCreated2DModel():
  inputs = Input((512,512,3))   #evtl anpassen
 
  conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
  pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
  
  conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
  pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
  
  conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(32, (1, 1), activation='relu', padding='same')(conv3)
  conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)
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
  pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)

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

  conv7 = Conv2D(1024, (1, 1), activation='relu', padding='same')(pool3)

  conv7 = Conv2D(1024, (1, 1), activation='relu', padding='same')(pool3)
  avgpool = GlobalAveragePooling2D()(conv7)


  
  # falsche aktivierungsfunktion
  outputs = Dense(4, activation='linear')(avgpool)

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
  pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)


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
  outputs = Dense(24, activation='sigmoid')(avgpool)

  model = Model(inputs=inputs, outputs=outputs)

  return model

model = getCreated2DModel()

print("vor compile")
# falsche loss funktion
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
print("nach compile")

print("vor fit")
#keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
#keras.callbacks.tensorboard_v1.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
filepath = "./model_"+(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")).replace(" ","_")+".h5"
callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=2)]
#generator = Generator(x_pathsTrainData, y_2D_labelTrainData, 16)
#generatorValidation = Generator(x_pathsValData, y_2D_labelValData, 16, validation=True)
#
#histGenerator = model.fit_generator(generator, validation_data=generatorValidation, steps_per_epoch=1000, epochs=10)
hist = model.fit(x_imgTrainData, y_2D_labelTrainData, validation_data = (x_imgValData, y_2D_labelValData), batch_size=32, epochs=20, callbacks=callbacks)

#
print("nach fit")
#
for cnt,key in enumerate(hist.history.keys()):
    
    plt.figure(999 + cnt)
    plt.plot(hist.history[key])
    plt.title(key.replace("_"," "))

#for cnt, key in enumerate(histGenerator.history.keys()):
#    plt.figure(999 + cnt)
#    plt.plot(histGenerator.history[key])
#    plt.title(key.replace("_"," "))

#-------------------
predImg = np.zeros(shape=(1,512,512,3))
img = cv2.resize(cv2.imread("./synthetic/000011.is.png"),(512,512))
predImg[0] = cv2.resize(img,(512,512))

parser = Parser("./synthetic/000011.json")
#-------------------

predictedData = model.predict(predImg)
predictedData[predictedData < 0] = 0

imgLabel = parser.get_2D_data()

print("myData: ", imgLabel)
print("myPredictedData: ", predictedData, " ", len(predictedData))


import matplotlib.patches as patches

fig,ax = plt.subplots(1)

ax.imshow(img)
yValue = predictedData.item(0)*512
xValue = predictedData.item(1)*512
width = predictedData.item(2)*512 - yValue
height = predictedData.item(3)*512 - xValue


yValueLabel = imgLabel[0]*512
xValueLabel = imgLabel[1]*512
widthLabel = imgLabel[2]*512 - yValueLabel
heightLabel = imgLabel[3]*512 - xValueLabel

rectLabel = patches.Rectangle((xValueLabel,yValueLabel),widthLabel,heightLabel,linewidth=1,edgecolor='b',facecolor='none')
ax.add_patch(rectLabel)

rectPred = patches.Rectangle((xValue,yValue),width,height,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rectPred)
plt.show()
