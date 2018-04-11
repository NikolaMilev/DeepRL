# name of the semaphore used for interprocess communication
SEM_NAME="/sem_deeprl"
# timeout for waiting on the semaphore -- currently unused
TIMEOUT=0.1

# name of the shared memory segment used for sharing game info (everything except for the screenshot)
SHM_NAME="/shm_deeprl"
# size of the aforementioned shared memory segment
SHM_SIZE=100

# name of the shared memory segment used for sharing the current screenshot of the game
SS_SHM_NAME="/ss_shm_deeprl"
# width of the screenshot
SS_W=400
# height of the screenshot
SS_H=300
# overall size of the screenshot memory segment 18 is the header, 3 is for R, G and B channels
SS_SHM_SIZE=SS_W*SS_H*3+18 # change, magic