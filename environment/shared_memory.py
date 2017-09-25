import posix_ipc as pi
import time
import os
import timeit

SEM_NAME="/sem_deeprl"
SHR_MEM=123321
SHM_NAME="/shm_deeprl"
SHM_SIZE=100
TIMEOUT=0.1

# 

def obtain_data(repeat=0):
	"""
		Try to set the semaphore, read the data from the shared memory and return the read byte array.
		We wait for TIMEOUT (can be float) seconds at most twice before giving up.
	"""
	sem = pi.Semaphore(SEM_NAME)
	sem.release()
	retval=None
	try:
		#print "Successfully opened semaphore!\n"
		#print "Waiting: "
		sem.acquire()
		retval=read_shm()
		#print "Posting: "
		sem.release()
		sem.close()
		#print "Closed!\n"
		if retval:
			return retval
		else:
			return None
	except pi.BusyError as e1:
		#print "BUSY"
		print e1
		if repeat == 1:
			return None
		else:
			return obtain_data(repeat=1)
	except Error as e:
		sem.release()
		sem.close()
		print e
		return None

def read_shm():
	"""
		Reads from the shared memory, without the semaphores. 
		Don't call this function without the semaphore, I haven't been playing around with file locks in python. 
	"""
	shm = pi.SharedMemory(SHM_NAME)
	r = os.read(shm.fd, SHM_SIZE).partition(b'\0')[0]
	a = [x.strip() for x in r.split(',')]
	#print a
	if len(a) != 4:
		return None
	os.close(shm.fd)
	return a[0], a[1], a[2], a[3]


# time.sleep(10)
# for i in range(1000):
# 	print obtain_data()
# 	time.sleep(0.1)
#num = 100000
#print timeit.timeit('print shared_memory.obtain_data()', number=num, setup="import shared_memory")/num
