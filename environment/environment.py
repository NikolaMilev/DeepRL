import utils
import PIL.Image as Image

class Environment:

	"""
		The complete environment state and means of communicating with it. 
		Things that I know should be done, thus far:
		 1) collect the screenshot DONE
		 2) preprocess it
		 3) send the signal to the game - TODO: keep it pressed for a short period of time?
		 4) collect the current score
		    something more?


		 Requirements (thus far collected):
		  - python-xlib (apt)
		  - pymouse (pip)
		  - everything needed to install SDL-Ball (link: http://sdl-ball.sourceforge.net/ )
		  - uinput (see utils module)
		 The purpose of this list is making a script to set up everything later on
	"""

	SS_SIZE = 80
	SS_COPY = None
	
	""" Although it's repeating inside the utils module, I find it a semantic part of the class. """
	ACTION_MOVE_LEFT = utils.ACTION_LEFT
	ACTION_MOVE_RIGHT = utils.ACTION_RIGHT
	ACTION_UP = utils.ACTION_UP


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
		return ss.resize((SS_SIZE, SS_SIZE), Image.ANTIALIAS).convert('LA')


	@classmethod
	def Action(cls, action):
		if(action < 0 or action >= utils.NUM_ACTIONS):
			raise ValueError('Action value must be 0, 1 or 2. See module environment.')
		
		utils.send_keystroke(action)