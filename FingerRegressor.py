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
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


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
	X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.02)
	return X_train, X_test, Y_train, Y_test



X_train, X_test, Y_train, Y_test = load_data()
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255


regressorInput = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
regressor = Regressor(regressorInput)
regressor.compile(optimizer = Adam(0.0005), loss= 'mean_squared_error', metrics=['mse'])
regressor.summary()


filepath='./RegressorModel/Regressor.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


batch_size = 64
epochs = 1000
history = regressor.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_test,Y_test), shuffle=True, verbose=1, callbacks=callbacks_list)
#Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./RegressorModel/regressorLoss.png')
#plt.show()
regressor.save_weights('./RegressorModel/Regressor-lastepoch.h5')
