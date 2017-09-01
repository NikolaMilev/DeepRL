import posix_ipc as pi
import time
import os
import timeit

# todo: check for all possible errors!
# for instance, if a semaphore doesn't exist, that means the process has not written to the shared memory or sth

SEM_NAME="/sem_deeprl"
SHR_MEM=123321
SHM_NAME="/shm_deeprl"
SHM_SIZE=100
TIMEOUT=0.1

def obtain_data(repeat=0):
	"""
		Try to set the semaphore, read the data from the shared memory and return the read byte array.
		We wait for TIMEOUT (can be float) seconds at most twice before giving up.
	"""
	sem = pi.Semaphore(SEM_NAME)
	retval=None
	try:
		#print "Successfully opened semaphore!\n"
		#print "Waiting: "
		sem.acquire(TIMEOUT)
		retval=read_shm()
		#print "Posting: "
		sem.release()
		sem.close()
		#print "Closed!\n"
		if retval != "" and retval != None:
			return retval
		else:
			return None
	except pi.BusyError:
		#print "BUSY"
		if repeat == 1:
			return None
		else:
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
	shm = pi.SharedMemory(SHM_NAME)
	r = os.read(shm.fd, SHM_SIZE).partition(b'\0')[0]
	a = [x.strip() for x in r.split(',')]
	#print a
	if len(a) != 3:
		return None
	os.close(shm.fd)
	return a[0], a[1], a[2]

#print obtain_data()
num = 100000
print timeit.timeit('print shared_memory.obtain_data()', number=num, setup="import shared_memory")/num
