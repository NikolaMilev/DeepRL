from environment import Environment as env
import utils
import timeit

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

DRL_PAUSED = 1
DRL_LVL_TRANS = 2
DRL_HIGHSCORE = 4
DRL_TITLE_SCREEN = 8
DRL_QUIT = 16


time.sleep(10)
while True:
	#ss = env.GetSS()
	a = shm.read_shm()
	if a:
		print a

	time.sleep(0.2)
	#utils.send_random_keystroke()





# s = timeit.timeit(stmt="a=utils.get_ss() ; b=utils.crop_center(a)", number=num_stmt, setup="import utils")
# print s/num_stmt

#s = timeit.timeit(stmt="os.system(\"xte 'key Up'\")", number = num_stmt, setup="import os")
#print s/num_stmt


# time.sleep(10)
# utils.send_keystroke('up')
# utils.send_keystroke('left')
#time.sleep(1)
#os.system("xte 'key Up'")
#os.spawnl(os.P_NOWAIT, "xte 'keydown Left' 'usleep 100000' 'keyup Left' ; echo pizda &")
#print "penis"
#os.system("xte 'keydown Right' 'usleep 100000' 'keyup Right' ")


#s = timeit.timeit(stmt="test.pizda()", number = num_stmt, setup="import os; import test")
#print s/num_stmt
