from environment import Environment as env
import utils
#import timeit
import random
import time
import shared_memory as shm


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



#a = env.GetSS()
#a.save("1.png")
# time.sleep(1)
# env.Action(2)

#a=Image.open('1.png')
#b=utils.crop_center(a)
#utils.get_all_bb(utils.crop_upper(b, 0.1))

# timeti returns number of seconds needed for the whole thing so we divide the result by the number of executions to get the average 
num_stmt = 1000

#utils.crop_center(utils.get_ss()).show()

def send_random():
	lista = ["up", "left", "right"]
	utils.send_keystroke(random.choice(lista))

#utils.send_name()

time.sleep(5)
while not env.quit():
	#ss = env.GetSS()
	print '---------------------------------------'
	env.update()
	env.get_back_in_game()
	print env.get_reward()
	print env.get_info()
	#print env.DSCORE, env.DLIVES
	send_random()
	time.sleep(0.2)
	#utils.send_random_keystroke()


