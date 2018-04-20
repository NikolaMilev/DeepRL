from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.utils import plot_model
from keras.backend import resize_images

from keras.models import model_from_json

# to be put to one configuration file
NET_W = 100
NET_H = 75
NET_D = 4
LEARNING_RATE = 1e-4

def buildNetwork():
	inp = Input(shape=(NET_H, NET_W, NET_D)) # the input layer
	conv_1 = Convolution2D(32, (8, 8), padding='same', activation='relu', strides=4)(inp) # the first convolutional layer
	conv_2 = Convolution2D(64, (4, 4), padding='same', activation='relu', strides=2)(conv_1) # the second convolutional layer
	conv_3 = Convolution2D(64, (3, 3), padding='same', activation='relu', strides=1)(conv_2) # the third convolutional layer
	flat = Flatten()(conv_3) 
	hidden = Dense(512, activation='relu')(flat) # the fully connected layer
	out = Dense(3, activation='softmax')(hidden) # the output layer

	model = Model(inputs=inp, outputs=out)
	adam = Adam(lr=LEARNING_RATE) # we specify the learning rate
	model.compile(loss='mse', optimizer=adam)
	
	print "Compiled the network!"
	return model

def saveNetwork(model):
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")

def loadNetwork():
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")
	return loaded_model