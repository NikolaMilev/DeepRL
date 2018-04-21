from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.utils import plot_model
from keras.backend import resize_images
from keras.models import model_from_json
from skimage import transform, color, io, exposure
import scipy.misc as sm
from collections import deque
import random
import os
import sys
import gym
import time

GAME="MsPacman-v0"
NET_W = 80
NET_H = 80
NET_D = 4
LEARNING_RATE = 1e-4

#MY_PATH=os.path.dirname(os.path.realpath(__file__))
#GAME_PATH=os.path.join("/home", "nmilev", "Desktop", "master", "SDLBall", "SDL-Ball_source_build_0006_src")

#sys.path.append("environment")

#from environment import Environment  as env

# to be put to one configuration file
#IMG_W = 400
#IMG_H = 300
#IMG_D = 3

#RESCALE_FACTOR=NET_W*1.0/IMG_W
REPLAY_MEMORY_SIZE = 50000

class ExperienceReplay:

	def __init__(self):
		self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)

	def add(self, item):
		self.memory.append(item)
	
	def randomSample(self, numitems):
		return random.sample(self.memory, numitems)


class DRLAgent():
	def __init__(self, envName):
		self.env = gym.make(envName)
		self.numActions = self.env.action_space.n
		self.er = ExperienceReplay()
		self.targetNetwork = DRLAgent.buildNetwork(numOutput=self.numActions)
		self.qNetwork = DRLAgent.buildNetwork(numOutput=self.numActions)

	@staticmethod
	def buildNetwork(numOutput, inputShape=(NET_H, NET_W, NET_D)):
		inp = Input(shape=inputShape) # the input layer
		conv_1 = Convolution2D(32, (8, 8), padding='same', activation='relu', strides=4)(inp) # the first convolutional layer
		conv_2 = Convolution2D(64, (4, 4), padding='same', activation='relu', strides=2)(conv_1) # the second convolutional layer
		conv_3 = Convolution2D(64, (3, 3), padding='same', activation='relu', strides=1)(conv_2) # the third convolutional layer
		flat = Flatten()(conv_3) 
		hidden = Dense(512, activation='relu')(flat) # the fully connected layer
		out = Dense(numOutput, activation='softmax')(hidden) # the output layer

		model = Model(inputs=inp, outputs=out)
		adam = Adam(lr=LEARNING_RATE) # we specify the learning rate
		model.compile(loss='mse', optimizer=adam)
		
		print("Compiled the network!")
		return model

	@staticmethod
	def saveNetwork(model):
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		saveWeights(model)
		print("Saved model to disk")
	
	@staticmethod
	def saveWeights(model):
		model.save_weights("model.h5")


	@staticmethod
	def loadNetwork():
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loadWeights(loaded_model)
		print("Loaded model from disk")
		return loaded_model

	@staticmethod
	def loadWeights(model):
		model.load_weights("model.h5")


	# TODO: check contrast/luminosity changing
	@staticmethod
	def preprocessImage(img):
		#x = img_to_array(img)
		x = img.astype(np.float64)
		x = x / 255.0
		x = color.rgb2gray(x)
		#x = exposure.equalize_adapthist(x, clip_limit=0.2) # adjust the image so that the tiles are not blended into the background
		x = transform.resize(x,(NET_H, NET_H))
		#print(x.shape)
		return x


	def run_episode(self):
		state = self.env.reset()
		done=None
		while done != True:
			self.env.render()
			x_t = self.preprocessImage(state)
			s_t = np.expand_dims(np.stack((x_t, x_t, x_t, x_t), axis=2), axis=0)
			y = self.qNetwork.predict(s_t)
			state, reward, done, info = self.env.step(np.argmax(y[0], axis=0))
		self.env.close()





def main():
	agent = DRLAgent(GAME)
	agent.run_episode()


main()


#model = neural_network.buildNetwork()
#img = load_img('/home/nmilev/Desktop/screenshot.tga')  # keras hardcodes PIL.Image.convert to mode "L", which is acting as a low-pass filter, truncating all values above 255
# #x = np.expand_dims(x, axis=0)
# #y = model.predict(x)
#x_t = preprocessImage(img)
# #print x_t
# y = model.predict(x_t)
# #plot_model(model, to_file="/home/nmilev/Desktop/model.png")
# #print y
#s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
#s_t = np.expand_dims(s_t, axis=0)
# # model = buildModel()
#y = model.predict(s_t)
# # print y
#sm.imsave("/home/nmilev/Desktop/jtzm.png", x_t)