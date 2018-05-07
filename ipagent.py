from keras.models import Model
import keras.backend as kback
from keras.layers import Input, Convolution2D, Dense, Flatten, Lambda, Multiply
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import numpy as np
from keras.models import model_from_json
from skimage import transform, color, io
import scipy.misc as sm
from collections import deque
import random
import gym
import datetime
import os
import time

import datetime
MODEL_NAME_APPEND=str(datetime.datetime.now())


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

COLAB=True

GAME="BreakoutDeterministic-v4"
NET_H = 105
NET_W = 80
NET_D = 4
LEARNING_RATE = 2.5e-4
MINIBATCH_SIZE=32
GAMMA=0.99 # discount factor
OBSERVE_MAX=30
SAVE_FREQ=10000
TRAIN_FREQ=4
TARGET_UPDATE_FREQ=10000
INITIAL_EPSILON=1.0
FINAL_EPSILON=0.1
EPSILON_EXPLORATION=1000000
NUM_EPISODES = 12000
INITIAL_REPLAY_MEMORY_SIZE=50000
MAX_REPLAY_MEMORY_SIZE=300000
DIRECTORY="colaboratory_models" if COLAB else "."
LOAD_NETWORK=None #os.path.join(DIRECTORY, "BreakoutDeterministic-v42018-05-05 11:13:40.906340.h5")
MODEL_FILENAME=GAME+MODEL_NAME_APPEND
PADDING="valid"
STATS_SAVE_FREQ = 10
MOMENTUM = 0.95  
MIN_GRAD = 0.01


def check(model):
	ws = model.get_weights()
	for x in ws:
		if not (np.isfinite(x)).all():
			return True
		if (np.isnan(x)).any():
			return True
	return False

def resolveReward(reward):
	return np.clip(reward, -1, 1)

class ExperienceReplay:
	"""
	deque items should be tuples: (s,a,r,s',t)
	where s is the current state, a is the action chosen, r is the reward, s' is the next state and 
	t is an indicator if the state s' is terminal
	every state is NET_D stacked screens, resized to size NET_H x NET_W
	"""
	def __init__(self):
		self.memory = deque(maxlen=MAX_REPLAY_MEMORY_SIZE)
		assert self.memory.maxlen == MAX_REPLAY_MEMORY_SIZE

	def size(self):
		return len(self.memory)

	def add(self, item):
		"""
		Adds the item to the experience replay. If the buffer already contains MAX_REPLAY_MEMORY_SIZE elements, the oldest
		is removed to make place for the new one.
		The item added must have 5 elements.
		The element with the index 0 must have shape (NET_H, NET_W, NET_D) and be of type uint8.
		The element with the index 3 must have shape (NET_H, NET_W, 1) and be of type uint8
		"""
		assert len(item) == 5
		assert item[0].shape == (NET_H, NET_W, NET_D)
		assert item[0].dtype == np.uint8
		assert item[3].shape == (NET_H, NET_W, NET_D)
		assert item[3].dtype == np.uint8

		self.memory.append(item)
	
	def randomSample(self, numitems, actionsCategorical=False, numCategorical=None):
		# assume that the number of categories is present if we should use them, otherwise we don't care
		assert numCategorical if actionsCategorical else True

		mb = random.sample(self.memory, numitems)

		prevStates = np.array([x[0] for x in mb])
		actions = to_categorical(np.array([x[1] for x in mb]), num_classes=numCategorical) if actionsCategorical else np.array([x[1] for x in mb])
		rewards = np.array([x[2] for x in mb])
		nextStates = np.array([x[3] for x in mb])
		terminals = np.array([x[4] for x in mb]).astype(dtype=bool)

		return (prevStates, actions, rewards, nextStates, terminals)

# vectorized huber mean
def huber_mean(yTrue, yPred):
	error = yTrue - yPred
	quad_term = error*error / 2.0
	lin_term = abs(error) - 0.5
	use_lin = kback.cast((abs(error) > 1.0), 'float32')
	return use_lin * lin_term + (1-use_lin) * quad_term


class DRLAgent():

	def __init__(self, envName):
		self.env = gym.make(envName)
		self.numActions = self.env.action_space.n
		self.er = ExperienceReplay()
		self.targetNetwork = DRLAgent.buildNetwork(numActions=self.numActions)
		self.qNetwork = DRLAgent.buildNetwork(numActions=self.numActions)
		if(LOAD_NETWORK):
			self.loadWeights(self.targetNetwork, LOAD_NETWORK)
			self.loadWeights(self.qNetwork, LOAD_NETWORK)
		else:
			self.updateWeights()
		self.timestep = 0
		self.previousEpisodeTimestep=0
		self.episode = 0
		self.duration = 0
		self.episodeReward = 0.0
		self.episodeLoss = 0.0
		self.epsilon = INITIAL_EPSILON
		self.epsilonStep = (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_EXPLORATION
		self.statsSaveFrequency = STATS_SAVE_FREQ
	
	def __del__(self):
		self.env.reset()
		self.env.close()

	def resetStats(self):
		self.duration = self.timestep - self.previousEpisodeTimestep
		self.previousEpisodeTimestep = self.timestep
		self.episodeLoss = 0.0
		self.episodeReward = 0.0

	def saveStats(self, path=os.path.join(DIRECTORY, MODEL_FILENAME+".csv")):
		with open(path, "a") as file:
			file.write("{},{},{:.4f},{},{},{:.4f}".format(self.episode, self.timestep, self.episodeLoss, self.episodeReward, self.duration, self.epsilon))

	def printStats(self):
		print("Ep: {}, Frame: {}, Loss: {:.4f}, Rew: {}, Dur: {}, Eps: {:.4f}, Mem size: {}".format(self.episode, self.timestep, self.episodeLoss, self.episodeReward, self.duration, self.epsilon, self.er.size()))

	@staticmethod
	def buildNetwork(numActions, inputShape=(NET_H, NET_W ,NET_D)):

		"""
		The network receives the state (stacked screenshots) and produces a vector that contains a 
		Q value for each possible action from that state 
		"""
		state_inp = Input(shape=inputShape) # the input layer
		action_inp = Input(shape=(numActions,)) # the action mask layer
		normalizer = Lambda(lambda x: x / 255.0)(state_inp) # the layer that divides all the input by 255 thus setting the values in [0,1]
		conv_1 = Convolution2D(32, (8, 8), padding=PADDING, activation='relu', strides=4)(normalizer) # the first convolutional layer
		conv_2 = Convolution2D(64, (4, 4), padding=PADDING, activation='relu', strides=2)(conv_1) # the second convolutional layer
		conv_3 = Convolution2D(64, (3, 3), padding=PADDING, activation='relu', strides=1)(conv_2) # the third convolutional layer
		flat = Flatten()(conv_3) 
		hidden = Dense(512, activation='relu')(flat) # the fully connected layer
		out = Dense(numActions)(hidden) # the output layer
		filtered_out = Multiply()([out, action_inp])
		model = Model(inputs=[state_inp, action_inp], outputs=filtered_out)
		optimizer = RMSprop(lr=LEARNING_RATE, rho=MOMENTUM, epsilon=MIN_GRAD)
		model.compile(loss=huber_mean, optimizer=optimizer)
		
		print("Compiled the network!")
		return model

	# Utility functions for saving and loading models and their weights
	@staticmethod
	def saveNetwork(model):
		model_json = model.to_json()
		with open(os.path.join(DIRECTORY, MODEL_FILENAME + ".json"), "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		DRLAgent.saveWeights(model)
		print("Saved model to disk")
	
	@staticmethod
	def saveWeights(model):
		model.save_weights(os.path.join(DIRECTORY, MODEL_FILENAME + ".h5"))
		print("Saved weights to disk")

	@staticmethod
	def saveWeightsPath(model, path):
		model.save_weights(path)

	@staticmethod
	def loadNetwork():
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		DRLAgent.loadWeights(loaded_model)
		print("Loaded model from disk")
		return loaded_model

	@staticmethod
	def loadWeights(model, path):
		model.load_weights(path)
		print("Loaded from {}".format(path))


	# image preprocessing
	# grayscale and downsample by 2 (half of both dimensions)
	@staticmethod
	def preprocessImage(img):
		x = np.mean(img, axis=2).astype(np.uint8)
		# downsampling to 105x80 (half the size on both dimensions)
		x =  x[::2, ::2]
		#print(x.shape)
		x= np.reshape(x, (NET_H, NET_W))
		#sm.toimage(x).save("/home/nmilev/Desktop/jtzm.png")
		return x		
	
	# since I'm not yet sure if set_weights and get_weights work, I'm saving to disk and then reading
	def updateWeights(self):
		"""
		Copy the weights from the online network to the target network.
		"""
		# print(np.array_equal(self.targetNetwork.get_weights(), self.qNetwork.get_weights()))
		#self.targetNetwork.set_weights(self.qNetwork.get_weights())
		DRLAgent.saveWeightsPath(self.qNetwork, "tmp.h5")
		DRLAgent.loadWeights(self.targetNetwork, "tmp.h5")
		
	def train(self):

		# as in the paper, we set:
		# yj = rj,											if the episode terminated at step j+1
		# yj = rj + GAMMA*max[a'](qTarget(next_states)), 	otherwise

		# obtain a batch (s, a, r, s', t) of MINIBATCH_SIZE elements from the experience replay memory
		states, actions, rewards, next_states, terminals = self.er.randomSample(MINIBATCH_SIZE, actionsCategorical=True, numCategorical=self.numActions)
		# we predict the Q values for next states for all actions, using target network
		qTargetValues = self.targetNetwork.predict([next_states, np.ones(actions.shape)], batch_size=MINIBATCH_SIZE)
		# for the terminal states, the Q value is 0 (for all actions)
		qTargetValues[terminals] = 0
			
		# Q(s,a) = R(s,a) + GAMMA*max[a'](Q'(s', a'))
		y = rewards + GAMMA * np.max(qTargetValues, axis=1)
		# multipy it element-wise with actions so that only the Q-values for the taken actions are updated
		y = np.expand_dims(y, axis=1) * actions
		self.episodeLoss += self.qNetwork.train_on_batch([states, actions], y)

		# check if there is infinity/nan weight in the network, a precaution
		if check(self.qNetwork) or check(self.targetNetwork):
			print("SOMETHING IS NAN/INF")

	# not sure if this is needed!
	# the purpose of this method is to prepare the state
	# if needed, it will return the max of the two screenshots
	# will examine in the future
	# TODO
	@staticmethod
	def prepState(previousObservation, observation):
		#x = np.maximum(DRLAgent.preprocessImage(previousObservation), DRLAgent.preprocessImage(observation))
		#or
		x = DRLAgent.preprocessImage(observation)
		#sm.toimage(x).save("/home/nmilev/Desktop/jtzm.png")
		#print("Saved")
		#return DRLAgent.preprocessImage(observation)
		return x

	@staticmethod
	def updateState(state, newObservation):
		x=np.append(state[:, :, 1:], np.reshape(newObservation, (NET_H, NET_W, 1)), axis=2)
		#print((state[:, :, 0] == state[:, :, 1]).all() and (state[:, :, 0] == state[:, :, 2]).all() and (state[:, :, 0] == state[:, :, 2]).all() and (state[:, :, 0] == state[:, :, 3]).all())
		return x

	# choose an action with epsilon greedy policy,
	# the first INITIAL_REPLAY_MEMORY_SIZE always being random
	# and getting the prediction from the targetNetwork otherwise
	def chooseAction(self, state):
		# with epsilon-greedy policy, we pick an action, the first INITIAL_REPLAY_MEMORY_SIZE always being random
		if self.epsilon >= random.random() or self.timestep < INITIAL_REPLAY_MEMORY_SIZE:
			return random.randrange(self.numActions)
		else:
			# otherwise, we ask the target network to predict the next step and act on it
			#print(np.expand_dims(np.ones(self.numActions), axis=0).shape)
			y = self.targetNetwork.predict([np.expand_dims(state, axis=0), np.expand_dims(np.ones(self.numActions), axis=0)])
			return np.argmax(y[0], axis=0)
			#print(y, retval)
		
	# learn for numEpisodes episodes
	def learn(self, numEpisodes=NUM_EPISODES, observationSteps=OBSERVE_MAX):
		self.timestep = 0
		
		# we learn for numEpisodes episodes
		for self.episode in range(numEpisodes):
			terminal = False
			observation = self.env.reset()
			# we keep a previous screenshot because we might want to add max of two consecutive screenshots
			# instead of one screenshot
			# TODO: how does this affect learning?
			previousObservation=None

			# for a random number from 0 to observationSteps, the agent stays still, i.e, it just observes
			for i in range(random.randint(1, observationSteps)):
				previousObservation = observation
				observation, _, _, _ = self.env.step(0)
				if not COLAB:
					self.env.render()
			
			# we obtain the initial screenshot
			statep = self.prepState(previousObservation, observation)
			# inital 4 screen state
			# the first 4 screens are the same but it will be updated over time
			state = np.stack([statep for _ in range(NET_D)], axis=2)
			while not terminal:
				if not COLAB:
					self.env.render()
				# again keeping the previous observation
				previousObservation = observation
				# we obtain the action; see the docs for the function for the manner of choosing
				action = self.chooseAction(state)
				# we step through the environment
				# not using the additional info as it may vary through games
				observation, reward, terminal, _ = self.env.step(action)
				# we prepare the state
				observation=self.prepState(previousObservation, observation)
				# we transform the reward so it fits the norm
				reward = resolveReward(reward)

				# we update the screenshot
				next_state = self.updateState(state, observation)
				# add the (s,a,r,s',t) 5-tuple to the experience replay
				self.er.add((state, action, reward, next_state, terminal))
				state = next_state
				# increment the reward and the timestep
				self.episodeReward += reward
				self.timestep += 1

				# we decrement the epsilon if needed but only after filling the initial
				# portion of the experience replay
				if self.epsilon > FINAL_EPSILON and self.timestep >= INITIAL_REPLAY_MEMORY_SIZE:
					self.epsilon -= self.epsilonStep

				self.doPeriodicStuff()

			# after an episode, we print the stats, skipping 
			if self.episode % self.statsSaveFrequency == 0:
				self.printStats()
				#self.saveStats()
			self.resetStats()


	# TODO: rename this function or move the code out of it
	def doPeriodicStuff(self):
		if self.timestep > INITIAL_REPLAY_MEMORY_SIZE:
			# if the inital memory replay is filled, we can start training
			if self.timestep % TRAIN_FREQ == 0:
				#print("Training!")
				self.train()
			if self.timestep % TARGET_UPDATE_FREQ == 0:
				print("Copying!")
				self.updateWeights()
			if self.timestep % SAVE_FREQ == 0:
				print("Saving!")
				self.saveWeights(self.targetNetwork)
		

	# TODO
	def runTest(self, numEpisodes):
		for i in range(numEpisodes):
			self.runEpisode()

	def runEpisode(self):
		"""
		Just for running an episode, not for anything else!
		TODO update it
		"""
		state = self.env.reset()
		done=None
		while done != True:
			if not COLAB:
				self.env.render()
			x_t = self.preprocessImage(state)
			s_t = np.stack([x_t for _ in range(NET_D)], axis=0)
			s_t = np.expand_dims(s_t, axis=0)
			y = self.targetNetwork.predict(s_t)
			state, reward, done, info = self.env.step(np.argmax(y, axis=1))
			print(info)


agent=DRLAgent(GAME)
agent.learn()