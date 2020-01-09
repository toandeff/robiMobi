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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from jsonParser import Parser
from generator import Generator
#from root_path + "generator.ipynb" import Generator

# how to extract data
#import zipfile
#zip_ref = zipfile.ZipFile(root_path + "/synthetic.zip", 'r')
#zip_ref.extractall("/tmp")
#zip_ref.close()

# train data and labels
directory_path = './synthetic/'
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
 
  conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
  pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
  
  conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
  pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
  
  conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(32, (1, 1), activation='relu', padding='same')(conv3)
  conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)
#  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#  pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
#  
#  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#  pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
#  
#  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#  conv3 = Conv2D(64, (1, 1), activation='relu', padding='same')(conv3)
#  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#  pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)
#
#  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#  conv4 = Conv2D(128, (1, 1), activation='relu', padding='same')(conv4)
#  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#  pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)
#
#  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
#  conv5 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv5)
#  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
#  conv5 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv5)
#  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
#  pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv5)
#
#  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
#  conv6 = Conv2D(512, (1, 1), activation='relu', padding='same')(conv6)
#  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)
#  conv6 = Conv2D(512, (1, 1), activation='relu', padding='same')(conv6)
#  conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)

#  conv7 = Conv2D(1024, (1, 1), activation='relu', padding='same')(pool3)

  conv7 = Conv2D(1024, (1, 1), activation='relu', padding='same')(pool3)
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
model.compile(optimizer='adam', loss='binary_crossentropy')
print("nach compile")
#generator = Generator(x_paths, y_2D_labelBox, 16) # yPaths brauchen wir nicht
#histGenerator = model.fit_generator(generator, steps_per_epoch=1000, epochs=10)
print("vor fit")
#keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
#keras.callbacks.tensorboard_v1.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
filepath="./"
callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)]
hist = model.fit(x_img, y_2D_labelBox, batch_size=16, epochs=100, callbacks=callbacks)
print("nach fit")

plt.figure(200)
plt.title("Loss")
plt.plot(hist.history["loss"])

#plt.figure(201)
#plt.title("val_loss")
#plt.plot(hist.history["val_loss"])


#
#x_img = np.zeros(shape=(len(x_paths),512,512,3))
#  x_img[idx] = cv2.resize(cv2.imread(x_paths[idx]),(512,512))
#  

#-------------------
predImg = np.zeros(shape=(1,512,512,3))
img = cv2.imread("./synthetic/000100.is.png")
predImg[0] = cv2.resize(img,(512,512))

parser = Parser("./synthetic/000100.json")
#-------------------

predictedData = model.predict(predImg)


imgLabel = parser.get_2D_data()

print("myData: ", imgLabel)
print("myPredictedData: ", predictedData, " ", len(predictedData))


import matplotlib.patches as patches

fig,ax = plt.subplots(1)

ax.imshow(img)
yValue = predictedData.item(0)*512
xValue = predictedData.item(1)*512
width = predictedData.item(2)*512-yValue
height = predictedData.item(3)*512-xValue


yValueLabel = imgLabel[0]*512
xValueLabel = imgLabel[1]*512
widthLabel = imgLabel[2]*512-yValueLabel
heightLabel = imgLabel[3]*512 - xValueLabel

rectLabel = patches.Rectangle((xValueLabel,yValueLabel),widthLabel,heightLabel,linewidth=1,edgecolor='b',facecolor='none')
ax.add_patch(rectLabel)

rectPred = patches.Rectangle((xValue,yValue),width,height,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rectPred)
plt.show()





predImg = np.zeros(shape=(1,512,512,3))
img = cv2.imread("./synthetic/000200.is.png")
predImg[0] = cv2.resize(img,(512,512))

parser = Parser("./synthetic/000200.json")
#-------------------

predictedData = model.predict(predImg)


imgLabel = parser.get_2D_data()

print("myData: ", imgLabel)
print("myPredictedData: ", predictedData, " ", len(predictedData))


import matplotlib.patches as patches

fig2,ax = plt.subplots(1)

ax.imshow(img)
yValue = predictedData.item(0)*512
xValue = predictedData.item(1)*512
width = predictedData.item(2)*512-yValue
height = predictedData.item(3)*512-xValue


yValueLabel = imgLabel[0]*512
xValueLabel = imgLabel[1]*512
widthLabel = imgLabel[2]*512-yValueLabel
heightLabel = imgLabel[3]*512 - xValueLabel

rectLabel = patches.Rectangle((xValueLabel,yValueLabel),widthLabel,heightLabel,linewidth=1,edgecolor='b',facecolor='none')
ax.add_patch(rectLabel)

rectPred = patches.Rectangle((xValue,yValue),width,height,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rectPred)
plt.show()


predImg_1 = np.zeros(shape=(1,512,512,3))
img_1 = cv2.imread("./synthetic/000001.is.png")
predImg_1[0] = cv2.resize(img_1,(512,512))

parser_1 = Parser("./synthetic/000001.json")
#-------------------

predictedData_1 = model.predict(predImg_1)


imgLabel_1 = parser_1.get_2D_data()

print("myData: ", imgLabel_1)
print("myPredictedData: ", predictedData_1, " ", len(predictedData_1))


import matplotlib.patches as patches

fig3,ax_1 = plt.subplots(1)

ax_1.imshow(img_1)
yValue_1 = predictedData_1.item(0)*512
xValue_1 = predictedData_1.item(1)*512
width_1 = predictedData_1.item(2)*512-yValue_1
height_1 = predictedData_1.item(3)*512-xValue_1


yValueLabel_1 = imgLabel_1[0]*512
xValueLabel_1 = imgLabel_1[1]*512
widthLabel_1 = imgLabel_1[2]*512-yValueLabel_1
heightLabel_1 = imgLabel_1[3]*512 - xValueLabel_1

rectLabel_1 = patches.Rectangle((xValueLabel_1,yValueLabel_1),widthLabel_1,heightLabel_1,linewidth=1,edgecolor='b',facecolor='none')
ax_1.add_patch(rectLabel_1)

rectPred_1 = patches.Rectangle((xValue_1,yValue_1),width_1,height_1,linewidth=1,edgecolor='r',facecolor='none')
ax_1.add_patch(rectPred_1)
plt.show()
#x_neuBilder
#for bilder in x_neuBilder:
#  (x,y, breite, hoehe) = hist.predict(x) 
#  cv2.drawRactangle(x,y,breite,hoehe)


#(x, y, breite, hoehe) = hist.predict(bild1)
#cv2.drawRactangle(x, y, breite,hoehe)