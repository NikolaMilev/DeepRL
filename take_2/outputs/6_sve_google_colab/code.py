#KERAS
from keras.models import Model
import keras.backend as kback
from keras.layers import Input, Convolution2D, Dense, Flatten, Lambda, Multiply
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.callbacks import History 

# for resizing
import scipy

#other
from skimage.transform import resize
import numpy as np
from collections import deque
import numpy as np
import gym
import random

import datetime
import os
# to make sure the image data is in the correct order
import keras.backend as backend
assert backend.image_data_format()=="channels_last"

# for frame testing purposes only
import matplotlib.pyplot as plt

GAME="BreakoutDeterministic-v4"
COLAB=True
USE_TARGET_NETWORK=True
SAVE_PATH=os.path.join("colaboratory_models", "colab_models") if COLAB else "."
SAVE_NAME=GAME+str(datetime.datetime.now())

NETWORK_UPDATE_FREQUENCY=10000 # in parameter updates, not in steps taken!

INITIAL_REPLAY_MEMORY_SIZE=50000
MAX_REPLAY_MEMORY_SIZE=1000000 if COLAB else 500000 # no memory in my own machine for full 1000000 frames so I go to half of that
OBSERVE_MAX=30
NUM_EPISODES = 20000 if COLAB else 50000 # refers to the number of in-game episodes, not learning episodes
# one learning episode is separated by loss of life 
MINIBATCH_SIZE=32
INITIAL_EPSILON=1.0
FINAL_EPSILON=0.1
EPSILON_DECAY_STEPS=1000000
GAMMA=0.99
# network details:
NET_H=84
NET_W=84
NET_D=4
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.95  
MIN_GRAD = 0.01
#LOSS=huberLoss


TRAIN_FREQUENCY=4
SAVE_FREQUENCY=10000
PADDING="valid"

INFO_WRITE_FREQ=10

# Utility functions
# I wish to keep this in one file so that I can use it from a notebook

history=History()

def huberLoss(a, b, inKeras=True):
	error = a - b
	quadraticTerm = error*error / 2
	linearTerm = abs(error) - 1/2
	useLinearTerm = (abs(error) > 1.0)
	if inKeras:
		useLinearTerm = backend.cast(useLinearTerm, 'float32')
	return useLinearTerm * linearTerm + (1-useLinearTerm) * quadraticTerm

LOSS=huberLoss

def buildNetwork(height, width, depth, numActions):
	state_in=Input(shape=(height, width, depth))
	action_in=Input(shape=(numActions, ))
	normalizer=Lambda(lambda x: x/255.0)(state_in)
	conv1=Convolution2D(filters=16, kernel_size=(8,8), strides=(4,4), padding=PADDING, activation="relu")(normalizer)
	conv2=Convolution2D(filters=32, kernel_size=(4,4), strides=(2,2), padding=PADDING, activation="relu")(conv1)
	
	flatten=Flatten()(conv2)
	dense=Dense(units=256, activation="relu")(flatten)
	out=Dense(units=numActions, activation="linear")(dense)
	filtered_out=Multiply()([out, action_in])
	model=Model(inputs=[state_in, action_in], outputs=filtered_out)
	opt=RMSprop(lr=LEARNING_RATE, rho=MOMENTUM, epsilon=MIN_GRAD)
	model.compile(loss=LOSS, optimizer=opt)

	print("Built and compiled the network!")
	return model

def copyModelWeights(srcModel, dstModel):
	dstModel.set_weights(srcModel.get_weights())

# the original
# def buildNetwork(height, width, depth, numActions):
# 	state_in=Input(shape=(height, width, depth))
# 	action_in=Input(shape=(numActions, ))
# 	normalizer=Lambda(lambda x: x/255.0)(state_in)
# 	conv1=Convolution2D(filters=32, kernel_size=(8,8), strides=(4,4), padding=PADDING, activation="relu")(normalizer)
# 	conv2=Convolution2D(filters=64, kernel_size=(4,4), strides=(2,2), padding=PADDING, activation="relu")(conv1)
# 	conv3=Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=PADDING, activation="relu")(conv2)
# 	flatten=Flatten()(conv3)
# 	dense=Dense(units=512, activation="relu")(flatten)
# 	out=Dense(units=numActions, activation="linear")(dense)
# 	filtered_out=Multiply()([out, action_in])
# 	model=Model(inputs=[state_in, action_in], outputs=filtered_out)
# 	opt=RMSprop(lr=LEARNING_RATE, rho=MOMENTUM, epsilon=MIN_GRAD)
# 	model.compile(loss=LOSS, optimizer=opt)

# 	print("Built and compiled the network!")
# 	return model

def saveModelWeights(model):
	savePath=os.path.join(SAVE_PATH, SAVE_NAME + ".h5")
	model.save_weights(savePath)
	print("Saved weights to {}".format(savePath))

def preprocessSingleFrameNew(img):
	view=img
	#view=img[::2,::2]
	x=(view[:,:,0]*0.299 + view[:,:,1]*0.587 + view[:,:,2]*0.114)
	p=scipy.misc.imresize(x, (84, 84)).astype(np.uint8)
	# plt.imshow(p)
	# plt.show()
	return p

def preprocessSingleFrame(img):
	# Y = 0.299 R + 0.587 G + 0.114 B
	# with double downsample
	#view = img[::2,::2]
	#return (view[:,:,0]*0.299 + view[:,:,1]*0.587 + view[:,:,2]*0.114).astype(np.uint8)
	return preprocessSingleFrameNew(img)

# we will use tuples!
def getNextState(state, nextFrame):
	return (state[1], state[2], state[3], preprocessSingleFrame(nextFrame))

def transformReward(reward):
	#return np.clip(reward, -1.0, 1.0)
	return reward

class ExperienceReplay():
	"""
	The class for memory. So far, one tuple is ~10KB
	I used tuples because this way, after preprocessing, up to 4 consecutive
	states share some frame data. When using numpy arrays, this was impossible
	as I was appending data. The only more efficient way is to use one large
	numpy array for all screenshots but this seems overly complicated because
	end of an episode edge case. If this turns out to be easy to implement, 
	I will reimplement this class.
	"""
	def __init__(self):
		self.memory=deque(maxlen=MAX_REPLAY_MEMORY_SIZE)
	def size(self):
		return len(self.memory)
	def addTuple(self, state, action, reward, nextState, terminal):
		self.memory.append((state, action, reward, nextState, terminal))
	def addItem(self, item):
		self.memory.append(item)
	def sample(self, sampleSize=MINIBATCH_SIZE):
		return random.sample(self.memory, sampleSize)
	def getMiniBatch(self, sampleSize=MINIBATCH_SIZE):
		minibatch=self.sample(sampleSize) # an array of tuples
		states=np.array([np.stack([frame for frame in tup[0]], axis=2) for tup in minibatch])
		actions=np.array([tup[1] for tup in minibatch])
		rewards=np.array([tup[2] for tup in minibatch])
		nextStates=np.array([np.stack([frame for frame in tup[3]], axis=2) for tup in minibatch])
		terminals=np.array([tup[4] for tup in minibatch])
		
		assert states.dtype==np.uint8
		assert nextStates.dtype==np.uint8
		assert terminals.dtype==bool

		return (states, actions, rewards, nextStates, terminals)




class DRLAgent():
	def __init__(self, envName):
		self.envName=envName
		self.env=gym.make(self.envName)
		self.numActions=self.env.action_space.n
		self.experienceReplay=ExperienceReplay()
		self.qNetwork=buildNetwork(NET_H, NET_W, NET_D, self.numActions)
		if(USE_TARGET_NETWORK):
			self.targetNetwork=buildNetwork(NET_H, NET_W, NET_D, self.numActions)
			copyModelWeights(srcModel=self.qNetwork, dstModel=self.targetNetwork)
		# actions chosen
		self.timeStep=0
		# episode count
		self.episodeCount=0
		# total episode reward
		self.episodeReward=0.0
		# total episode loss; has nothing to do with reward as the loss
		# is obtained by training on random batches
		self.episodeLoss=0.0
		# the total episode duration in frames, including no-op frames
		self.episodeDuration=0

		# the initial epsilon (exploration) value
		self.epsilon=INITIAL_EPSILON
		# the value by which epsilon is decreased for every action taken after
		# the INITIAL_REPLAY_MEMORY_SIZEth frame
		self.epsilonDecay=(INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY_STEPS

		self.parameterUpdates=0 # to count the number of parameter updates so I can copy network weights


	def printInfo(self):
		print("Ep: {}, Dur: {}, Step: {}, Rew: {:.2f}, Loss: {:.4f}, Eps: {:.4f}, Mem.size: {}".format(self.episodeCount, self.episodeDuration, self.timeStep, self.episodeReward, self.episodeLoss, self.epsilon, self.experienceReplay.size()))

	def chooseAction(self, state):
		retval=None
		if self.timeStep < INITIAL_REPLAY_MEMORY_SIZE or np.random.rand() < self.epsilon:
			retval=self.env.action_space.sample()
		else:
			stacked_state=np.stack(state, axis=2)
			if(USE_TARGET_NETWORK):
				y=self.targetNetwork.predict([np.expand_dims(stacked_state, axis=0), np.expand_dims(np.ones(self.numActions), axis=0)])
			else:
				y=self.qNetwork.predict([np.expand_dims(stacked_state, axis=0), np.expand_dims(np.ones(self.numActions), axis=0)])
			retval=np.argmax(y, axis=1)
		assert retval!=None

		if self.epsilon > FINAL_EPSILON and self.timeStep >= INITIAL_REPLAY_MEMORY_SIZE:
			self.epsilon-=self.epsilonDecay
		return retval

	def trainOnBatch(self, batchSize=MINIBATCH_SIZE):
		self.parameterUpdates+=1
		states, actions, rewards, nextStates, terminals=self.experienceReplay.getMiniBatch()
		actions=to_categorical(actions, num_classes=self.numActions)
		if(USE_TARGET_NETWORK):
			nextStateValues=self.targetNetwork.predict([nextStates, np.ones(actions.shape)], batch_size=batchSize)
		else:
			nextStateValues=self.qNetwork.predict([nextStates, np.ones(actions.shape)], batch_size=batchSize)
		assert terminals.dtype==bool
		nextStateValues[terminals]=0
		# 
		y=rewards + GAMMA * np.max(nextStateValues, axis=1)
		y=np.expand_dims(y, axis=1)*actions
		#self.episodeLoss += self.qNetwork.train_on_batch([states, actions], y)

		hist=self.qNetwork.fit([states, actions], y, batch_size=batchSize, epochs=1, verbose=0)
		self.episodeLoss+=np.mean(hist.history['loss'])

		if(USE_TARGET_NETWORK and self.parameterUpdates % NETWORK_UPDATE_FREQUENCY == 0):
			copyModelWeights(srcModel=self.qNetwork, dstModel=self.targetNetwork)
			print("Updated target network!")

		
	def learn(self, numEpisodes=NUM_EPISODES):
		self.timeStep=0
		for self.episodeCount in range(numEpisodes):
			self.episodeDuration=0
			self.episodeLoss=0
			self.episodeReward=0
			self.episodeDuration=0

			terminal=False
			observation=self.env.reset() # return frame

			for _ in range(random.randint(1, OBSERVE_MAX)):
				observation, _, _, info=self.env.step(0)
				self.episodeDuration += 1
			curLives=info['ale.lives']
			frame=preprocessSingleFrame(observation)
			state=(frame, frame, frame, frame)
			nextState=None
			while not terminal:
				action=self.chooseAction(state)
				observation, reward, terminal, info = self.env.step(action)
				newLives=info['ale.lives']
				# I found that the loss of life means the end of an episode
				if newLives < curLives:
					terminalToInsert=True
				else:
					terminalToInsert=False

				curLives=newLives
				nextState=getNextState(state, observation)
				# I wish to see the raw reward
				self.episodeReward+=reward
				reward=transformReward(reward)
				self.experienceReplay.addTuple(state, action, reward, nextState, terminalToInsert)
				
				if self.experienceReplay.size() >= INITIAL_REPLAY_MEMORY_SIZE:
					if self.timeStep % TRAIN_FREQUENCY == 0:
						self.trainOnBatch()
					if self.timeStep % SAVE_FREQUENCY == 0:
						saveModelWeights(self.qNetwork)

				self.timeStep+=1
				self.episodeDuration += 1
				
				
				state=nextState

			if self.episodeCount % INFO_WRITE_FREQ == 0:
				self.printInfo()
				# plt.imshow(nextState[0])
				# plt.show()