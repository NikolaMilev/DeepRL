# from environment import Environment as env
import utils
import timeit
# import random
import time
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
