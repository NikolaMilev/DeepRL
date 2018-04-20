from collections import deque
import random

# to be put to one configuration file
REPLAY_MEMORY = 50000

class ExperienceReplay:

	def __init__(self):
		self.memory = deque(maxlen=REPLAY_MEMORY)

	def add(self, item):
		self.memory.append(item)
	
	def randomSample(self, numitems):
		return random.sample(self.memory, numitems)