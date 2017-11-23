#Importing Libraries
import os
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
#from scipy import misc
from scipy.misc import imread
#from matplotlib import pyplot as plt
import PIL
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

#File containing the path to images and the labels [path/to/images label]
filename = './train.txt'
filenameV = './val.txt'
filenameT = './test.txt'

#Lists where to store the paths and labels
filenames = []
labels = []
filenamesV = []
labelsV = []
filenamesT = []


#Reading file and extracting paths and labels
with open(filename, 'r') as File:
    infoFile = File.readlines() #Reading all the lines from File
    for line in infoFile: #Reading line-by-line
        words = line.split() #Splitting lines in words using space character as separator
        filenames.append(words[0])
        labels.append(int(words[1]))

with open(filenameV, 'r') as File:
    infoFile = File.readlines() #Reading all the lines from File
    for line in infoFile: #Reading line-by-line
        words = line.split() #Splitting lines in words using space character as separator
        filenamesV.append(words[0])
        labelsV.append(int(words[1]))

with open(filenameT, 'r') as File:
    infoFile = File.readlines() #Reading all the lines from File
    for line in infoFile: #Reading line-by-line
        filenamesT.append(line.strip())


NumFilesT = len(filenamesT)
#NumLabelsV = len(labelsV)
print(NumFilesT)
#print(NumLabelsV)


print(type(filenamesT))
#list
#print(type(labelsV))
#list

#print(filenames)
#print(labels)

print(filenamesT[0])
img = Image.open(filenamesT[0])
#img = img.convert('L')
img = img.resize((64,64),PIL.Image.LANCZOS)
#img.show()
#print(type(img))

img=np.asarray(img,np.float32)
#print(img)
print(type(img))
print(img.shape)

temp = []
for image_path in filenames:
    img = Image.open(image_path)
    img = img.resize((64,64),PIL.Image.LANCZOS)
    img = np.asarray(img,np.float32)
    temp.append(img)

print(len(temp))
print(type(temp))

train_x = np.stack(temp)
print(len(train_x))
print(type(train_x))
print(train_x.shape)

train_x /= 255.0

print(len(train_x))
print(type(train_x))
print(train_x.shape)
#print(train_x)

tempV = []
for image_path in filenamesV:
    img = Image.open(image_path)
    img = img.resize((64,64),PIL.Image.LANCZOS)
    img = np.asarray(img,np.float32)
    tempV.append(img)

val_x = np.stack(tempV)
print(len(val_x))
print(type(val_x))
print(val_x.shape)

val_x /= 255.0

print(len(val_x))
print(type(val_x))
print(val_x.shape)
#print(val_x)

tempT = []
for image_path in filenamesT:
    img = Image.open(image_path)
    img = img.resize((64,64),PIL.Image.LANCZOS)
    img = np.asarray(img,np.float32)
    tempT.append(img)

test_x = np.stack(tempT)
print(len(test_x))
print(type(test_x))
print(test_x.shape)

test_x /= 255.0

print(len(test_x))
print(type(test_x))
print(test_x.shape)
#print(test_x)


train_y = np_utils.to_categorical(labels, 5)
print(train_y)
print(train_y.shape)

val_y = np_utils.to_categorical(labelsV, 5)
print(val_y)
print(val_y.shape)




model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', data_format="channels_last", input_shape=(64,64,3)))
print(model.output_shape)
# 
model.add(Conv2D(32, (3, 3), activation='relu', data_format="channels_last"))
print(model.output_shape)
# 
model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
print(model.output_shape)
# 
model.add(Dropout(0.25))
print(model.output_shape)
# 


model.add(Conv2D(64, (3, 3), activation='relu', data_format="channels_last", input_shape=(64,64,3)))
print(model.output_shape)
# 
model.add(Conv2D(64, (3, 3), activation='relu', data_format="channels_last"))
print(model.output_shape)
# 
model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
print(model.output_shape)
# 
model.add(Dropout(0.25))
print(model.output_shape)
# 


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y, 
          batch_size=16, epochs=20, verbose=1)
# Epoch 1/20
# 1744/2569 [==>...........................] - ETA: 96s - loss: 0.5806 - acc: 0.8164

# Evaluate model on test data
score = model.evaluate(val_x, val_y, verbose=0)

print(score)
print(model.metrics_names)


test_y = model.predict(test_x, batch_size=16, verbose=0)

pred_y = np.argmax(test_y,axis=1)
print("pred_y")
print(type(pred_y))
print(pred_y)
print(pred_y.shape)

np.savetxt('project2_55768.txt', pred_y, fmt='%10.0f')






