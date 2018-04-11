# supposed to do all the preprocessing needed before the image goes into the neural network


def process(image):
	return adjustColor(resize(cropCenter(image)))

# is it needed?
# rescale better
def cropCenter(image):
	return image
def resize(image):
	return image
def adjustColor(image):
	return image