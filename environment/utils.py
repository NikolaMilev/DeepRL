import gtk.gdk
import PIL.Image as Image

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
