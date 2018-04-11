import utils
import PIL.Image as Image
import shared_memory as shm
import random
import RewardResolver
import ImagePreprocessor
"""
	I shall be using the class as a singleton
"""
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
	# the default game values for normal mode
	DEFAULT_VALUES = (0, 3, 0)

	# all the constants
	SS_SIZE = 80
		# the game info constants
	DRL_PAUSED = 1
	DRL_DEAD = 2
	DRL_HIGHSCORE = 4
	DRL_TITLE_SCREEN = 8
	DRL_QUIT = 16

	# highscore screen variable; 1 if we have sent the player name before, 0 otherwise
	REACHED_TITLESCREEN = 0

	IMAGE=None

	# variables
	SCR = None
	# number of lives
	LIVES=0
	# life count change
	DLIVES=0
	
	# current level (goes from 0)
	LEVEL=0
	# level change
	DLEVEL=0
	#game info variable; bitwise conjunction with one of the DRL_ constans indicates
	# a certain game stage
	GAME_INFO=0

	"""
		One of the methods to check if a certain game stage has been reached.
		This one checks if the game is paused.
	"""
	@classmethod
	def paused(cls):
		if cls.GAME_INFO & cls.DRL_PAUSED:
			return True
		else:
			return False

	"""
		One of the methods to check if a certain game stage has been reached.
		If the player is dead (no more lives).  
	"""
	@classmethod
	def dead(cls):
		if cls.GAME_INFO & cls.DRL_DEAD:
			return True
		else:
			return False

	"""
		One of the methods to check if a certain game stage has been reached.
		This one checks if the highscore screen is shown. 
	"""
	@classmethod
	def highscore(cls):
		if cls.GAME_INFO & cls.DRL_HIGHSCORE:
			return True
		else:
			return False

	"""
		One of the methods to check if a certain game stage has been reached.
		This one checks if the title screen is shown. 
	"""
	@classmethod
	def title_screen(cls):
		if cls.GAME_INFO & cls.DRL_TITLE_SCREEN:
			return True
		else:
			return False


	"""
		One of the methods to check if a certain game stage has been reached.
		This one checks if the game is quit (not likely to happen but still.) 
	"""
	@classmethod
	def quit(cls):
		if cls.GAME_INFO & cls.DRL_QUIT:
			return True
		else:
			return False


	"""
		One of the methods to check if a certain game stage has been reached.
		This one checks if we are in game. This is the only stage where we take action
		and take screenshots.
	"""
	@classmethod
	def in_game(cls):
		return not cls.GAME_INFO

	@classmethod
	def send_name(cls):
		if not cls.REACHED_TITLESCREEN:
			utils.send_name()
			cls.REACHED_TITLESCREEN = 1
		else:
			utils.send_keystroke("enter")
	@classmethod
	def unpause(cls):
		utils.send_keystroke("esc")
	
	@classmethod
	def exit_title_screen(cls):
		utils.send_keystroke("menu")



	@classmethod
	def get_back_in_game(cls):
		if cls.highscore():
			cls.send_name()
		if cls.title_screen():
			cls.exit_title_screen()
		if cls.paused():
			#cls.unpause()
			pass
		

	@classmethod
	def get_info(cls):
		return str(cls.GAME_INFO) + " " + str(RewardResolver.getReward()) + " " + str(cls.LIVES) + " " + str(cls.LEVEL) 

	@classmethod
	def get_img(cls):
		return cls.IMAGE

	"""
		The interface for action sending
	"""
	@classmethod
	def receive_action(cls, action):
		utils.send_keystroke(action)

	@classmethod
	def get_reward(cls):
		return RewardResolver.getReward()
	


	"""
		Reset the environment variables
	"""
	@classmethod
	def reset(cls):
		cls.SCR = None
		cls.LIVES=cls.DEFAULT_VALUES[1]
		cls.DLIVES=0
		RewardResolver.reset()
		cls.LEVEL=cls.DEFAULT_VALUES[2]
		cls.DLEVEL=0
		#cls.GAME_INFO=0
		cls.IMAGE=None

	@classmethod
	def update(cls):
		# read shared memory
		data = shm.obtain_data()
		if(data):
			cls.IMAGE=ImagePreprocessor.process(data[1])
			# obtain the game info (see DRL_ constants)
			cls.GAME_INFO=int(data[0][0])
			# if we are dead, we reset the variables
			if cls.dead():
				cls.reset()
				return

			# obtain new score and give it to the RewardResolver
			RewardResolver.updateScore(data[0][1])

			#obtain new life count but first save the change in DLIVES
			cls.DLIVES=int(data[0][2])-cls.LIVES
			cls.LIVES=int(data[0][2])

			#obtain new level count but first save the change in DLEVEL
			cls.DLEVEL=int(data[0][3])-cls.LEVEL
			cls.LEVEL=int(data[0][3])

	
