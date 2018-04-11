import gtk.gdk
import PIL.Image as Image
from subprocess import call
import numpy
import os
import random
import time

keystrokes = {"up":"Up", "left":"Left", "right":"Right", "enter":"Return", "menu":"F1", "esc":"Escape"}
NUM_ACTIONS = len(keystrokes)

# current version uses a call to xte. used timeit on it and it's not slow. The only thing that could be a problem is the wait time
# but the process is in the background so no worries

def send_keystroke(key=None, wait=50000):
	if key in keystrokes:
		os.system("xte 'keydown {}' 'usleep {}' 'keyup {}' &".format(keystrokes[key], wait, keystrokes[key]))
	else:
		print "{}: keystroke not found".format(key)

def send_random_keystroke():
	send_keystroke(key=random.choice(keystrokes.keys()))

# sending the name in the high score menu -- expected to be used A LOT :)
def send_name():
	time.sleep(1)
	os.system("xte 'keydown Shift_L' 'keydown D' 'keyup D' 'keydown R' 'keyup R' 'keydown L' 'keyup L' 'keyup Shift_L' ")
	time.sleep(1)
	os.system("xte 'keydown Return' 'keyup Return'")