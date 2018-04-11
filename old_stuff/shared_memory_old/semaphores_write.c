#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <string.h>
#include <assert.h>

#define SEM_NAME "/sem_deeprl"
#define SHM_NAME "/shm_deeprl"
#define SHM_SIZE 51
#define MODE (S_IRUSR | S_IWUSR | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH)

int shm_send_message(char* msg)
{
	sem_t* sem;
	sem = sem_open(SEM_NAME, O_CREAT, S_IWUSR | S_IRUSR, 1);
	int fd;
		
	void *addr;

	

	if(sem == SEM_FAILED)
	{
		printf("failure semaphore\n");
		return -1;
	}
	sem_wait(sem);

	fd = shm_open(SHM_NAME, O_CREAT | O_RDWR | O_TRUNC, S_IRUSR | S_IWUSR);
	if (fd == -1)
	{
		perror("shm_open");
		return -1;
	}
	if (ftruncate(fd, SHM_SIZE) == -1)
	{
		perror("ftruncate");
		return -1;
	}

	addr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (addr == MAP_FAILED)
	{
		perror("mmap");
		return -1;
	}

	strcpy(addr, msg);

	close(fd);
	sleep(5);
	sem_post(sem);


	sem_close(sem);

	return 0;
}

int main(int argc, char* argv[])
{
	assert(argc == 2);
	shm_send_message(argv[1]);

	

	return 0;
}