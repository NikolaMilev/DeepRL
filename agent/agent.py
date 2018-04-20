import neural_network
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform, color, io, exposure
import scipy.misc as sm
import numpy as np


# to be put to one configuration file
IMG_W = 400
IMG_H = 300
IMG_D = 3
NET_W = 100
NET_H = 75
RESCALE_FACTOR=NET_W*1.0/IMG_W

# TODO: check contrast/luminosity changing
def preprocessImage(img):
	x = img_to_array(img)
	x = x / 255.0
	x = color.rgb2gray(x)
	#x = exposure.adjust_log(x, 1000000)
	x = exposure.equalize_adapthist(x, clip_limit=0.2) # adjust the image so that the tiles are not blended into the background
	x = transform.resize(x,(IMG_H*RESCALE_FACTOR,IMG_W*RESCALE_FACTOR))
	print x.shape
	return x

#model = neural_network.buildNetwork()
img = load_img('/home/nmilev/Desktop/screenshot.tga')  # keras hardcodes PIL.Image.convert to mode "L", which is acting as a low-pass filter, truncating all values above 255
# #x = np.expand_dims(x, axis=0)
# #y = model.predict(x)
x_t = preprocessImage(img)
# #print x_t
# y = model.predict(x_t)
# #plot_model(model, to_file="/home/nmilev/Desktop/model.png")
# #print y
s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

s_t = np.expand_dims(s_t, axis=0)
# # model = buildModel()
#y = model.predict(s_t)
# # print y
#sm.imsave("/home/nmilev/Desktop/jtzm.png", x_t)