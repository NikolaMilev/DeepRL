import posix_ipc as pi
import time
import os
import io
import timeit
from PIL import Image
import configuration as conf
import time

SEM_NAME=conf.SEM_NAME

SHM_NAME=conf.SHM_NAME
SHM_SIZE=conf.SHM_SIZE

SS_SHM_NAME=conf.SS_SHM_NAME
SS_W=conf.SS_W
SS_H=conf.SS_H
# 18 is the header, 3 is for R, G and B channels
SS_SHM_SIZE=conf.SS_SHM_SIZE

class SemShm():
	def __init__(self):
		try:
			# try to make a new semaphore
			# the flags pi.O_CREAT | pi.O_EXCL indicate that we should create a new one and it fails if 
			# there already exists one with the same name so we unlink it and make a new one
			self.sem = pi.Semaphore(SEM_NAME, flags=pi.O_CREAT | pi.O_EXCL, initial_value=1)
		except pi.ExistentialError as e:
			# 
			self.sem = pi.Semaphore(SEM_NAME)
			self.sem.unlink()
			self.sem = pi.Semaphore(SEM_NAME, flags=pi.O_CREAT | pi.O_EXCL, initial_value=1)
	
	def lock(self):
		self.sem.acquire()
	def unlock(self):
		self.sem.release()
	def getValue(self):
		return self.sem.value
	def obtain_data(self):
		"""
			Try to set the semaphore, read the data from the shared memory and return the read byte array.
			We wait for [default timeout] seconds before giving up.
		"""	
		retval=None
		retimg=None

		self.lock()
		time.sleep(0.05)
		retval=self.read_shm_game_info()
		retimg=self.read_shm_screenshot()
		print "Posting: "
		self.unlock()
		print "Closed!\n"
		if retval and retimg:
			return retval,retimg
		else:
			if retimg:
				retimg.close()
			return None


	@staticmethod
	def read_shm_game_info():
		"""
			Reads from the shared memory, without the semaphores. 
			Don't call this function without the semaphore, I haven't been playing around with file locks in python. 
		"""
		shm = None
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

	@staticmethod
	def read_shm_screenshot():
		"""
			Reads image from the shared memory, without the semaphores. 
			Don't call this function without the semaphore, I haven't been playing around with file locks in python. 
		"""
		ind=0
		try:
			shm = pi.SharedMemory(SS_SHM_NAME)
			r = os.read(shm.fd, SS_SHM_SIZE)
			os.close(shm.fd)
			ind=1
			bytes = bytearray(r)
			image = Image.open(io.BytesIO(bytes))
			return image
		except:
			if shm and ind == 0:
				os.close(shm.fd)
			print "-------------RETURNING NONE IMAGE-----------"
			return None

