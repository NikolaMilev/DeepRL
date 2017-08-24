from environment import Environment as env
import utils
import timeit
import random
import time
#env.GetSS().show()
#env.Action(0)

#a=utils.get_ss()
#b=utils.crop_center(a)
#b.show()

# timeti returns number of seconds needed for the whole thing so we divide the result by the number of executions to get the average 
# num_stmt = 1000
# s = timeit.timeit(stmt="a=utils.get_ss() ; b=utils.crop_center(a)", number=num_stmt, setup="import utils")
# print s/num_stmt

# for i in range(30):
# 	p = random.randrange(utils.NUM_ACTIONS)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)
# 	env.Action(p)

# 	time.sleep(1)