import scipy

	p=scipy.misc.imresize(x, (84, 84)).astype(np.uint8)
	plt.imshow(p)
	plt.show()