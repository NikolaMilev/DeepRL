import utils
import PIL.Image as Image
import shared_memory as shm

class Environment:

	"""
		The complete environment state and means of communicating with it. 
		Things that I know should be done, thus far:
		 1) collect the screenshot DONE
		 2) preprocess it
		 3) send the signal to the game - TODO: keep it pressed for a short period of time?
		 4) collect the current score, lives, lvl (level transition is not always or ever rewarded)
		 	Dead information
		    something more?


		 Requirements (thus far collected):
		  - python-xlib (apt) : to be deleted
		  - pymouse (pip) : to be deleted
		  - everything needed to install SDL-Ball (link: http://sdl-ball.sourceforge.net/ )
		  - uinput (see utils module)
		 The purpose of this list is making a script to set up everything later on
	"""
	SS_SIZE = 80
	DRL_PAUSED = 1
	DRL_DEAD = 2
	DRL_HIGHSCORE = 4
	DRL_TITLE_SCREEN = 8
	DRL_QUIT = 16

	
	SCR = None
	LIVES=0
	DLIVES=0
	SCORE=0
	DSCORE=0
	LEVEL=0
	DLEVEL=0
	GAME_INFO=0
	
	

	@classmethod
	def reset(cls):
		SCR = None
		LIVES=0
		SCORE=0
		GAME_INFO=0

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
		return utils.crop_center(ss)


	@classmethod
	def Action(cls, action):
		if(action < 0 or action >= utils.NUM_ACTIONS):
			raise ValueError('Action value must be 0, 1 or 2. See module environment.')
		
		utils.send_keystroke(action)

	@classmethod
	def update(cls):
		data = shm.read_shm()
		print data
		if(data):


			cls.GAME_INFO=int(data[0])

			cls.DSCORE=int(data[1])-cls.SCORE
			cls.SCORE=int(data[1])

			cls.DLIVES=int(data[2])-cls.LIVES
			cls.LIVES=int(data[2])

			cls.DLEVEL=int(data[3])-cls.LEVEL
			cls.LEVEL=int(data[3])
