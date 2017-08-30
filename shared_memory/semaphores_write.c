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

int main(int argc, char* argv[])
{
	// sem_t* sem;
	// sem = sem_open(SEM_NAME, O_CREAT, S_IWUSR | S_IRUSR, 1);
	key_t key;
	char *shm, *s;
	int fd;
		
	void *addr;

	assert(argc == 2);

	// if(sem == SEM_FAILED)
	// {
	// 	printf("failure semaphore\n");
	// 	return 0;
	// }
	// printf("Successfully opened semaphore!\n");
	// printf("Waiting: %d\n", sem_wait(sem));
	// sleep(5);
	// printf("Posting: %d\n", sem_post(sem));
	// sem_close(sem);
	// printf("Successfully closed semaphore!\n");

	fd = shm_open(SHM_NAME, O_CREAT | O_RDWR | O_TRUNC, S_IRUSR | S_IWUSR);
	if (fd == -1)
	{
		perror("shm_open");
		return 0;
	}
	if (ftruncate(fd, SHM_SIZE) == -1)
	{
		perror("ftruncate");
		return 0;
	}

	addr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (addr == MAP_FAILED)
	{
		perror("mmap");
		return 0;
	}

	strcpy(addr, argv[1]);

	close(fd);

	return 0;
}