'''
Author: Rishi Sharma
'''

import os, sys
import cv2
from keras.layers.core import *
from keras.layers import  Input,Dense,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import scipy
import numpy.random as rng
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from math import sqrt


def Regressor(input_img):
	reg_conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block1_conv1")(input_img)
	reg_conv1_1 = BatchNormalization()(reg_conv1_1)
	reg_conv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same',  name = "block1_conv2")(reg_conv1_1)
	reg_conv1_2 = BatchNormalization()(reg_conv1_2)
	reg_pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(reg_conv1_2)

	reg_conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1")(reg_pool1)
	reg_conv2_1 = BatchNormalization()(reg_conv2_1)
	reg_conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2")(reg_conv2_1)
	reg_conv2_2 = BatchNormalization()(reg_conv2_2)
	reg_pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(reg_conv2_2)

	reg_conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1")(reg_pool2)
	reg_conv3_1 = BatchNormalization()(reg_conv3_1)
	reg_conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2")(reg_conv3_1)
	reg_conv3_2 = BatchNormalization()(reg_conv3_2)
	reg_pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(reg_conv3_2)

	reg_flat = Flatten()(reg_pool3)
	fc1 = Dense(256, activation='relu')(reg_flat)
	fc2 = Dense(64, activation='relu')(fc1)
	fc3 = Dense(16, activation='relu')(fc2)
	fc4 = Dense(2, activation='relu')(fc3)
	regress = Model(inputs = input_img, outputs =  fc4)
	return regress

new_shape1 = 320
new_shape2 = 256


def load_data():
	imagePath = "/media/biometric/Data21/Core_Point/data_used/"
	maskPath = "/media/biometric/Data21/Core_Point/Mask_gt/"
	gtPath = "/media/biometric/Data21/Core_Point/gTruth/"
	imageExt = ".bmp"
	gtExt = "_gt.txt"
	maskExt = ".bmp"
	files = []
	if os.path.exists(imagePath) and os.path.exists(maskPath) and os.path.exists(gtPath):
		files = os.listdir(imagePath)
	else:
		sys.exit("Invalid Path")
	images = []
	masks = []
	gt = []
	for file in files:
		filename = file.split('.')[0]
		imagefile = imagePath+filename+imageExt
		maskfile = maskPath+filename+maskExt
		gtfile = gtPath+filename+gtExt

		if not(os.path.exists(imagefile)) or not(os.path.exists(maskfile)) or not(os.path.exists(gtfile)):
			continue

		im = cv2.imread(imagefile,0)
		original_shape1, original_shape2 = im.shape
		im = cv2.resize(im, (new_shape2,new_shape1))
		images.append(im)
		im = cv2.imread(maskfile,0)
		im = cv2.resize(im, (new_shape2,new_shape1))
		masks.append(im)

		f = open(gtfile, 'r')
		y, x = map(float, f.readline().split())
		x = (x*new_shape1)/original_shape1
		y = (y*new_shape2)/original_shape2
		gt.append((x,y))
	images = np.array(images)
	masks = np.array(masks)
	y = np.array(gt)
	x = np.stack((images,masks), axis = -1)
	return x, y, images


X_test, Y_test, images = load_data()
print(X_test.shape)
print(Y_test.shape)

X_test = X_test.astype('float32')
X_test/=255

filepath='./RegressorModel/Regressor.h5'
regressorInput = Input((X_test.shape[1], X_test.shape[2], X_test.shape[3]))
regressor = Regressor(regressorInput)
regressor.compile(optimizer = Adam(0.0005), loss= 'mean_squared_error', metrics=['mse', 'acc'])
regressor.summary()
regressor.load_weights(filepath)

f = open("./RegressorModel/Difference.txt", "w")

distances = []

for i in range(X_test.shape[0]):
	image = X_test[i].reshape(1,X_test[i].shape[0],X_test[i].shape[1],X_test[i].shape[2])
	x, y = regressor.predict(image)[0]
	x = abs(x-Y_test[i][0])
	y = abs(y-Y_test[i][1])
	d = sqrt(x*x + y*y)
	distances.append(d)
	f.write(str(x) + " " + str(y) + " "  + str(d)+"\n")
plt.hist(distances)
distances = np.array(distances)
m = np.mean(distances)
stdev = np.std(distances)
n, bins, patches = plt.hist(x=distances, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85, log=True)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Distances')
plt.ylabel('Frequency')
plt.title('Testing on Training Data')
plt.text(23, 45, r'$\mu=' + str(m) + ', b=' + str(stdev) +'$')
maxfreq = n.max()
plt.savefig('./RegressorModel/test.png')

f.close()
