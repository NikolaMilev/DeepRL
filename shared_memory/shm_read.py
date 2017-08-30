import posix_ipc as pi
import time
import os

SHM_NAME="/shm_deeprl"
SHM_SIZE=51

shm = pi.SharedMemory(SHM_NAME)
print os.read(shm.fd, SHM_SIZE)