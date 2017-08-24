import utils
import PIL.Image as Image
from pymouse import PyMouse

class Environment:

	"""
		The complete environment state and means of communicating with it. 
		Things that I know should be done, thus far:
		 1) collect the screenshot
		 2) preprocess it
		 3) send the signal to the game
		 4) something more?


		 Requirements (thus far collected):
		  - python-xlib (apt)
		  - pymouse (pip)

		 The purpose of this list is making a script to set up everything.
	"""

	SS_SIZE = 80
	SS_COPY = None
	NUM_ACTIONS = 3
	ACTION_MOVE_LEFT = 0
	ACTION_MOVE_RIGHT = 1
	ACTION_CLICK = 2


	@classmethod
	def GetSS(cls):
		""" 
			Obtain a screenshot, return a SS_SIZE x SS_SIZE grayscale representation of it; also saves it internally for further use, non-resized; 
			whether to always use it as grayscale is perhaps to be modified.
			Seems to work fine when the VM is minimized.
		
		"""
		ss = utils.get_ss()
		#global SS_COPY
		cls.SS_COPY = ss.copy()
		return ss.resize((80, 80), Image.ANTIALIAS).convert('LA')


	@classmethod
	def Action(cls, action):
		if(action < 0 or action > 2):
			raise ValueError('Action value must be 0, 1 or 2. See module environment.')
		
		m = PyMouse()
		x_dim, y_dim = m.screen_size()
		m.drag(0, 0)
		pass
