import gtk.gdk
import PIL.Image as Image
from subprocess import call
import cv2
import numpy
""" 
uinput module; pip install python-uinput; link: http://tjjr.fi/sw/python-uinput/ 
For uinput, you need to add permissions to the current user to write to /dev/uinput 
 """
import uinput

NUM_ACTIONS = 3
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_UP = 2
MAGIC_RATIO = 1.3

num_to_act = {ACTION_LEFT:uinput.KEY_LEFT, ACTION_RIGHT:uinput.KEY_RIGHT, ACTION_UP:uinput.KEY_UP}

# perhaps add sudo?
call(['modprobe', 'uinput'])

# after the creation, there must be a small delay; if there is any trouble, have this in mind
device = uinput.Device([
	uinput.KEY_UP, 
	uinput.KEY_LEFT, 
	uinput.KEY_RIGHT
	])



def crop_center(img):
	width,height = img.size
	#print x,y
	side = min(width,height)
	#print side

	new_width = int(side*MAGIC_RATIO)
	new_height = side

	left = (width - new_width)/2
	top = (height - new_height)/2
	right = (width + new_width)/2
	bottom = (height + new_height)/2

	return img.crop((left, top, right, bottom))

def get_ss():
	""" Returns a snapshot of the screen as a PIL image """
	w = gtk.gdk.get_default_root_window()
	sz = w.get_size()
	#print "The size of the window is %d x %d" % sz
	pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,False,8,sz[0],sz[1])
	pb = pb.get_from_drawable(w,w.get_colormap(),0,0,0,0,sz[0],sz[1])

	height=pb.get_height()
	width=pb.get_width()
	im = Image.frombuffer("RGB", (width,height) ,pb.pixel_array, 'raw', 'RGB', 0, 1)
	im.transpose(Image.FLIP_TOP_BOTTOM)
	return im

# needs to be redone with evdev
def send_keystroke(action):
	""" Sends the signal corresponding to je action value """
	device.emit_click(num_to_act[action])
