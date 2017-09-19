#include <iostream>
#include <cstdio> 
#include <string>
#include <sstream>
#include <cstdlib>

#include <unistd.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <string.h>
#include <assert.h>


// #include <sys/stat.h>

// g++ shared_memory.cpp -lpthread -lrt



#define SEM_NAME "/sem_deeprl"
#define SHM_NAME "/shm_deeprl"
#define SHM_SIZE 100
#define MODE (S_IRUSR | S_IWUSR | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH)

// resolved, to be tested
unsigned DRL_PAUSED = 0; // if the game is paused 1
unsigned DRL_LVL_TRANS = 0; // if the game is in the state of level transition 2
unsigned DRL_HIGHSCORE = 0; // when the game enters the high score table 4
unsigned DRL_TITLE_SCREEN = 0; // if the game is in the title screen 8 
unsigned DRL_QUIT = 0; // if we quit the game  16


void close_all(sem_t* sem, int fd, const char* msg)
{
	if(sem != NULL)
	{
		sem_close(sem);
	}
	if(fd >= 0)
	{
		close(fd);
	}
	perror(msg);
}

int shm_send_message(const char* msg)
{
	if(strlen(msg) == 0)
	{
		return 1;
	}
	sem_t* sem;
	sem = sem_open(SEM_NAME, O_CREAT, S_IWUSR | S_IRUSR, 1);
	int fd = -1;
		
	void *addr;

	

	if(sem == SEM_FAILED)
	{
		close_all(sem, fd, "failure semaphore");
		return -1;
	}
	sem_wait(sem);

	fd = shm_open(SHM_NAME, O_CREAT | O_RDWR | O_TRUNC, S_IRUSR | S_IWUSR);
	if (fd == -1)
	{
		close_all(sem, fd, "shm_open");
		return -1;
	}
	if (ftruncate(fd, SHM_SIZE) == -1)
	{
		close_all(sem, fd, "ftruncate");
		return -1;
	}

	addr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (addr == MAP_FAILED)
	{
		close_all(sem, fd, "mmap");
		return -1;
	}

	strcpy((char*)addr, msg);

	close(fd);
	sem_post(sem);


	sem_close(sem);

	return 0;
}

int shm_send_score(unsigned ind, int score, int lives)
{
	std::stringstream stream;
    stream << ind << "," << score << "," << lives;
	//std::cout << stream.str() << std::endl;
	return shm_send_message(stream.str().c_str());
}

int shm_send_game_data(int score, int lives)
{
	return shm_send_score(DRL_PAUSED | (DRL_LVL_TRANS << 1) | (DRL_HIGHSCORE << 2) | (DRL_TITLE_SCREEN << 3) | (DRL_QUIT << 4), score, lives);
}

// int main(int argc, char* argv[])
// {
// 	assert(argc == 4);
// 	shm_send_score(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
// 	// int i;
// 	// for(i = 0; i < 10000; i++)
// 	// {
// 	// 	shm_send_message(argv[1]);
// 	// 	//sleep(1);
// 	// }

	

// 	return 0;
// }