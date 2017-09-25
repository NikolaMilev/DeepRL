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


#utils.send_name()

time.sleep(5)
for i in xrange(100):
	#ss = env.GetSS()
	print '---------------------------------------'
	env.update()

	b=env.GAME_INFO
	

	if not b:
		print "in game"
	if b & env.DRL_PAUSED:
		print "paused"
		#utils.send_keystroke("esc")
	if b & env.DRL_DEAD:
		print "MUERTO"
	if b & env.DRL_HIGHSCORE:
		print "highscore!"
		time.sleep(3)
		utils.send_name()
	if b & env.DRL_TITLE_SCREEN:
		print "title screen"
		#utils.send_keystroke("menu")
	if b & env.DRL_QUIT:
		print "end"
		break

	#print env.DSCORE, env.DLIVES
	time.sleep(1)
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
