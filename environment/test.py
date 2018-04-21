from environment import Environment as env
import utils
#import timeit
import random
import time
#from shared_memory import SemShm
from PIL import Image
import io
import os
import posix_ipc as pi
MY_PATH=os.path.dirname(os.path.realpath(__file__))
GAME_PATH=os.path.join("/home", "nmilev", "Desktop", "master", "SDLBall", "SDL-Ball_source_build_0006_src")


"""
	TODO:
		Check if in menu or if game has finished;
		Menu checking can be done with var.menu, as far as I see.
		Finished-checking can be done just before the return from the main function
		Perhaps high score to write the name?
		Anything else?
		Perhaps, for the best results, find ONE place to write the number indicating the stage of the game, the score and the number of lives
"""


# from PIL import Image


#utils.crop_center(utils.get_ss()).show()

def send_random():
	lista = ["up", "left", "right"]
	utils.send_keystroke(random.choice(lista))

#utils.send_name()


img=None
env.startGame()
# while not env.quit():
# 	print '---------------------------------------'
# 	before = int(round(time.time() * 1000))
# 	env.update()
# 	env.get_back_in_game()
# 	print env.get_reward()
# 	print env.get_info()
# 	img=env.get_img()
# 	after = int(round(time.time() * 1000))
# 	# print "Time: ", (after-before)

# 	# if img:
# 	# 	print "Ima slike"
# 	# 	try:
# 	# 		pass
# 	# 		#img.save("/home/nmilev/Desktop/screenshot.tga")
# 	# 	except:
# 	# 		print "Img not saved, jbgy"
# 	# else:
# 	# 	print "Image None"
	
# 	#print env.DSCORE, env.DLIVES
# 	send_random()
# 	time.sleep(0.05)
# 	#utils.send_random_keystroke()
# img.save("/home/nmilev/Desktop/screenshot.tga")
# print "Saved the image!"

while(True):
	x_t, r_t, t_t = env.step()
	print r_t, t_t
	time.sleep(0.05)
