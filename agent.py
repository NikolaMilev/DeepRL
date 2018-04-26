from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Flatten
from keras.optimizers import Adam
import numpy as np
from keras.models import model_from_json
from skimage import transform, color, io
import scipy.misc as sm
from collections import deque
import random
import gym


GAME="MsPacman-v0"
NET_W = 70
NET_H = 70
NET_D = 4
LEARNING_RATE = 0.00025
MINIBATCH_SIZE=32
GAMMA=0.99 # discount factor
OBSERVE_MAX=30
SAVE_FREQ=10000
TRAIN_FREQ=4
TARGET_UPDATE_FREQ=10000
INITIAL_EPSILON=1.0
FINAL_EPSILON=0.1
EPSILON_EXPLORATION=100000
NUM_EPISODES = 12000
INITIAL_REPLAY_MEMORY_SIZE=5000
MAX_REPLAY_MEMORY_SIZE=30000
LOAD_NETWORK=True


def resolveReward(reward):
	return np.clip(reward, -1, 1)

class ExperienceReplay:
	"""
	deque items should be tuples: (s,a,r,s',t)
	where s is the current state, a is the action chosen, r is the reward, s' is the next state and 
	t is an indicator if the state s' is terminal
	every state is NET_D stacked screens, resized to size NET_W x NET_H
	"""
	def __init__(self):
		self.memory = deque(maxlen=MAX_REPLAY_MEMORY_SIZE)

	def add(self, item):
		self.memory.append(item)
		#if len(self.memory) % 1000 == 0:
		#	print("ExperienceReplay size: {}".format(len(self.memory)))
	
	def randomSample(self, numitems):
		return random.sample(self.memory, numitems)


class DRLAgent():

	def __init__(self, envName):
		self.env = gym.make(envName)
		self.numActions = self.env.action_space.n
		self.er = ExperienceReplay()
		self.targetNetwork = DRLAgent.buildNetwork(numOutput=self.numActions)
		self.qNetwork = DRLAgent.buildNetwork(numOutput=self.numActions)
		self.timestep = 0
		self.episode = 0
		self.epsilon = INITIAL_EPSILON
		self.epsilonStep = (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_EXPLORATION
		self.totalReward = 0
		self.totalLoss = 0
	def __del__(self):
		self.env.reset()
		self.env.close()

	@staticmethod
	def buildNetwork(numOutput, inputShape=(NET_D, NET_W, NET_H)):

		"""
		The network receives the state (stacked screenshots) and produces a vector that contains a 
		Q value for each possible action from that state 
		"""

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
		print("Saved weights to disk")


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
		x = img.astype(np.float64)
		x = x / 255.0
		x = color.rgb2gray(x)
		#x = exposure.equalize_adapthist(x, clip_limit=0.2) # adjust the image so that the tiles are not blended into the background
		x =  transform.resize(x,(NET_W, NET_H))
		return np.reshape(x, (NET_W, NET_H))


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
			self.env.render()
			x_t = self.preprocessImage(state)
			s_t = np.stack([x_t for _ in range(NET_D)], axis=0)
			#print(s_t.shape)
			s_t = np.expand_dims(s_t, axis=0)
			y = self.targetNetwork.predict(s_t)
			state, reward, done, info = self.env.step(np.argmax(y, axis=1))

		
	def updateWeights(self):
		# print(np.array_equal(self.targetNetwork.get_weights(), self.qNetwork.get_weights()))
		self.targetNetwork.set_weights(self.qNetwork.get_weights())
		#print("Updated weights!")
		
	def train(self):
		mb = self.er.randomSample(MINIBATCH_SIZE)
		states = np.array([x[0] for x in mb])
		actions = np.array([x[1] for x in mb])
		rewards = np.array([x[2] for x in mb])
		next_states = np.array([x[3] for x in mb])
		not_terminals = np.array([0 if a else 1 for a in [x[4] for x in mb]]) # 0 if the state is terminal and 1 otherwise
		#print("STATES SHAPE:")
		#print(states.shape)
		targets = self.targetNetwork.predict(next_states, batch_size=MINIBATCH_SIZE)
		expected_q = self.targetNetwork.predict(next_states, batch_size=MINIBATCH_SIZE)
		targets[range(MINIBATCH_SIZE), actions] = rewards + not_terminals * GAMMA * np.max(expected_q, axis=1)
		#print("Training on batch")
		self.totalLoss += self.qNetwork.train_on_batch(states, targets)

	# not sure if this is needed!
	@staticmethod
	def prepState(prev_obs, observation):
		x = np.maximum(DRLAgent.preprocessImage(prev_obs), DRLAgent.preprocessImage(observation))
		#sm.toimage(x).save("/home/nmilev/Desktop/jtzm.png")
		#print("Saved")
		return x

	@staticmethod
	def updateState(state, newObservation):
		return np.append(state[1:, :, :], np.reshape(newObservation, (1, NET_W, NET_H)), axis=0)

	def chooseAction(self, state):
		if self.epsilon >= random.random() or self.timestep < INITIAL_REPLAY_MEMORY_SIZE:
			retval = random.randrange(self.numActions)
		else:
			y = self.targetNetwork.predict(np.expand_dims(state, axis=0))
			retval = np.argmax(y[0], axis=0)

		if self.epsilon > FINAL_EPSILON and self.timestep >= INITIAL_REPLAY_MEMORY_SIZE:
			self.epsilon -= self.epsilonStep

		return retval

	def learn(self, numEpisodes=NUM_EPISODES, observationSteps=OBSERVE_MAX):
		self.timestep = 0
		if(LOAD_NETWORK):
			self.loadWeights(self.targetNetwork)
			self.loadWeights(self.qNetwork)
			
		for _ in range(numEpisodes):
			terminal = False
			observation = self.env.reset()
			for i in range(random.randint(1, observationSteps)):
				prev_obs = observation
				observation, _, _, _ = self.env.step(0)
				self.env.render()
			statep = self.prepState(prev_obs, observation)
			# inital 4 screen state
			# the first 4 screens are the same but it will be updated over time
			state = np.stack([statep for _ in range(NET_D)], axis=0)
			while not terminal:
				#print("New choice!")
				prev_obs = observation
				act = self.chooseAction(state)
				observation, reward, terminal, _ = self.env.step(act)
				self.env.render()
				state = self.run(state, act, reward, terminal, self.prepState(prev_obs, observation))
			print("Total reward: {}, Total loss: {}".format(self.totalReward, self.totalLoss))



	def run(self, state, action, reward, terminal, envObservation):
		next_state = self.updateState(state, envObservation)
		reward = resolveReward(reward)
		self.totalReward += reward
		
		self.er.add((state, action, reward, next_state, terminal))

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
			
			

			# first, we observe for OBSERVE_MAX timesteps
			# then, we begin the training process
			# every SAVE_FREQ timesteps, we save the network

		self.timestep = self.timestep + 1

		return next_state

def main():
	agent = DRLAgent(GAME)
	agent.learn()
	#agent.runTest(3)

main()

