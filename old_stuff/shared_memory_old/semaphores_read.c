#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <string.h>

#define SEM_NAME "/sem_deeprl"
#define SHM_NAME "/shm_deeprl"
#define SHM_SIZE 51
#define MODE (S_IRUSR | S_IWUSR | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH)

int main()
{
	// sem_t* sem;
	// sem = sem_open(SEM_NAME, O_CREAT, S_IWUSR | S_IRUSR, 1);
	int i;
	char *shm, *s;
	int fd;
	void *addr;
	char a[SHM_SIZE];
	struct stat sb;
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

	if (fstat(fd, &sb) == -1)
	{
		perror("fstat");
		return 0;
	}

	addr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (addr == MAP_FAILED)
	{
		perror("mmap");
		return 0;
	}

	write(STDOUT_FILENO, addr, sb.st_size);
	printf("\n");
	return 0;
}