import posix_ipc as pi
import time
import os
import io
import timeit
from PIL import Image

SEM_NAME="/sem_deeprl"

SHM_NAME="/shm_deeprl"
SHM_SIZE=100
TIMEOUT=0.1

SS_SHM_NAME="/ss_shm_deeprl"
SS_SHM_SIZE=800*600*3+18 # change, magic

# 

def obtain_data(repeat=0):
	"""
		Try to set the semaphore, read the data from the shared memory and return the read byte array.
		We wait for TIMEOUT (can be float) seconds at most twice before giving up.
	"""
	sem = pi.Semaphore(SEM_NAME)
	sem.release()
	retval=None
	retimg=None
	try:
		print "Successfully opened semaphore!\n"
		print "Waiting: "
		sem.acquire()
		retval=read_shm()
		retimg=read_shm_img()
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
	except pi.BusyError as e1:
		#print "BUSY"
		print e1
		if repeat == 1:
			return None
		else:
			sem.release()
			sem.close()
			return obtain_data(repeat=1)
	except:
		sem.release()
		sem.close()
		return None

def read_shm():
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


def read_shm_img():
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

