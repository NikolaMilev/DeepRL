import posix_ipc as pi
import time
import os
import io
import timeit
from PIL import Image
import configuration as conf

SEM_NAME=conf.SEM_NAME

SHM_NAME=conf.SHM_NAME
SHM_SIZE=conf.SHM_SIZE
TIMEOUT=conf.TIMEOUT

SS_SHM_NAME=conf.SS_SHM_NAME
SS_W=conf.SS_W
SS_H=conf.SS_H
# 18 is the header, 3 is for R, G and B channels
SS_SHM_SIZE=conf.SS_SHM_SIZE


def obtain_data():
	"""
		Try to set the semaphore, read the data from the shared memory and return the read byte array.
		We wait for [default timeout] seconds before giving up.
	"""
	try:
		sem = pi.Semaphore(SEM_NAME)
	except pi.ExistentialError as piee:
		print "Semaphore does not exist!"
		return None
	
	retval=None
	retimg=None
	try:
		print "Successfully opened semaphore!\n"
		print "Waiting: "
		sem.acquire()
		retval=read_shm_game_info()
		retimg=read_shm_screenshot()
		print "Posting: "
		sem.release()
		sem.close()
		print "Closed!\n"
		if retval and retimg:
			return retval,retimg
		else:
			if retimg:
				retimg.close()
			return None
	except:
		print "Error reading from shared memory!"
		sem.release()
		sem.close()
		return None

def read_shm_game_info():
	"""
		Reads from the shared memory, without the semaphores. 
		Don't call this function without the semaphore, I haven't been playing around with file locks in python. 
	"""
	try:
		shm = pi.SharedMemory(SHM_NAME)
		r = os.read(shm.fd, SHM_SIZE).partition(b'\0')[0]
		os.close(shm.fd)
		a = [x.strip() for x in r.split(',')]
		#print a
		if len(a) != 4:
			return None
		
		return a[0], a[1], a[2], a[3]
	except:
		if shm:
			os.close(shm.fd)
		return None


def read_shm_screenshot():
	"""
		Reads image from the shared memory, without the semaphores. 
		Don't call this function without the semaphore, I haven't been playing around with file locks in python. 
	"""
	try:
		shm = pi.SharedMemory(SS_SHM_NAME)
		r = os.read(shm.fd, SS_SHM_SIZE)
		os.close(shm.fd)
		bytes = bytearray(r)
		image = Image.open(io.BytesIO(bytes))
		return image
	except:
		if shm:
			os.close(shm.fd)
		return None

