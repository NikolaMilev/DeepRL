#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/shm.h>
#include <string.h>

#define SEM_NAME "/sem_deeprl"
#define SHM_KEY 123321
#define SHM_SIZE 51

int main()
{
	// sem_t* sem;
	// sem = sem_open(SEM_NAME, O_CREAT, S_IWUSR | S_IRUSR, 1);
	key_t key;
	char *shm, *s;
	int shmid;
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

	if ((shmid = shmget(SHM_KEY, SHM_SIZE, IPC_CREAT | 0666)) < 0) {
        perror("shmget");
        return 0;
    }

    if ((shm = shmat(shmid, NULL, 0)) == (char *) -1) {
        perror("shmat");
        return 0;
    }

    printf("\n%s\n", shm);

	return 0;
}