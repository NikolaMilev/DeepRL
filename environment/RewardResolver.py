# used for reward manipulation
# should have these features (API):
#	-- update the reward, giving the latest delta
#
# purpose of this module is to have the reward transformation logic in one place

latestScore=None
latestDelta=None

def reset():
	latestScore=None
	latestDelta=None

def updateScore(newReward):
	global latestScore, latestDelta
	if latestScore:
		latestDelta = int(newReward) - latestScore
		latestScore = int(newReward)
	else:
		latestScore=int(newReward)
		latestDelta=int(newReward)
	return transformReward(latestDelta)

def getReward():
	global latestDelta
	return transformReward(latestDelta)

# this is the most primitive way: if the the score delta is positive, reward is 1, if the score delta is
# negative, the reward is -1 and the reward is 0 otherwise.
def transformReward(reward):
	if reward > 0:
		return 1
	elif reward < 0:
		return -1
	else:
		return 0