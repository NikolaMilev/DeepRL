import posix_ipc as pi
import time
import os

# todo: check for all possible errors!
# for instance, if a semaphore doesn't exist, that means the process has not written to the shared memory or sth

SEM_NAME="/sem_deeprl"
SHR_MEM=123321
SHM_NAME="/shm_deeprl"
SHM_SIZE=100
def obtain_data():
	retval=None
	sem = pi.Semaphore(SEM_NAME)
	print "Successfully opened semaphore!\n"
	print "Waiting: "
	sem.acquire()
	retval=read_shm()
	print "Posting: "
	sem.release()
	sem.close()
	print "Closed!\n"
	if retval != "" and retval != None:
		return retval
	else:
		return None

def read_shm():
	shm = pi.SharedMemory(SHM_NAME)
	r = os.read(shm.fd, SHM_SIZE).partition(b'\0')[0]
	a = [x.strip() for x in r.split(',')]
	if len(a) != 3:
		return None
	return a[0], a[1], a[2]
	
print obtain_data()