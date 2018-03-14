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
#include <time.h>


// #include <sys/stat.h>

// g++ shared_memory.cpp -lpthread -lrt



#define SS_W 800
#define SS_H 600
#define SS_HEADER_SIZE 18

#define SEM_NAME "/sem_deeprl"
#define SHM_NAME "/shm_deeprl"
#define SHM_SIZE 100
#define SS_SHM_SIZE SS_W*SS_H*3 + SS_HEADER_SIZE
#define MODE (S_IRUSR | S_IWUSR | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH)

// mislim nepotrebno, koristicu isti semafor jer saljem podatke+ss u isto vreme
#define SS_SEM_NAME "/ss_sem_deeprl"
#define SS_SHM_NAME "/ss_shm_deeprl"

unsigned char screenshot_buffer[SS_SHM_SIZE];

// resolved, to be tested
unsigned DRL_PAUSED = 0; // if the game is paused 1
unsigned DRL_HIGHSCORE = 0; // when the game enters the high score table 4
unsigned DRL_TITLE_SCREEN = 0; // if the game is in the title screen 8 
unsigned DRL_QUIT = 0; // if we quit the game  16

// unresolved
// Perhaps will not be needed
unsigned DRL_DEAD = 0; // if you're dead

sem_t* sem;
int fd ;

int shm_send_buffer_no_sem(const void* buffer, size_t buf_size, const char* shm_name, size_t shm_size);
void close_all(const char* msg)
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


int shm_send_score(unsigned ind, int score, int lives, int level)
{
	std::stringstream stream;
    stream << ind << "," << score << "," << lives << "," << level;
	//std::cout << stream.str() << std::endl;
	return shm_send_buffer_no_sem(stream.str().c_str(), strlen(stream.str().c_str())+1, SHM_NAME, SHM_SIZE);
}

int shm_send_buffer_no_sem(const void* buffer, size_t buf_size, const char* shm_name, size_t shm_size)
{
	if(buffer == NULL || buf_size == 0)
	{
		return -1;
	}
	// da li?
	fd = -1;
		
	void *addr;

	fd = shm_open(shm_name, O_CREAT | O_RDWR | O_TRUNC, S_IRUSR | S_IWUSR);
	if (fd == -1)
	{
		std::cout << "SHM open failed!" << std::endl;
		return -1;
	}
	if (ftruncate(fd, shm_size) == -1)
	{
		std::cout << "SHM truncate failed!" << std::endl;
		return -1;
	}

	addr = mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (addr == MAP_FAILED)
	{
		std::cout << "mmap failed!" << std::endl;
		return -1;
	}

	memcpy(addr, buffer, buf_size);
	
	if (munmap(addr, shm_size) == -1)
	{
		std::cout << "Unmap failed!" << std::endl;
		return -1;
    }

	close(fd);
	return 0;
}

int shm_send_game_data(int score, int lives, int level, const void* buffer, size_t size)
{
	// semaphore open
	int ind;
	sem = sem_open(SEM_NAME, O_CREAT, S_IWUSR | S_IRUSR, 1);
	if(sem == SEM_FAILED)
	{
		//close_all(sem, -1, "failure semaphore");
		return -1;
	}
	if(sem_wait(sem) < 0)
	{
		return -1;
	}

	// write game info
	ind = shm_send_score(DRL_PAUSED | (DRL_DEAD << 1) | (DRL_HIGHSCORE << 2) | (DRL_TITLE_SCREEN << 3) | (DRL_QUIT << 4), score, lives, level);
	if(ind < 0)
	{
		close_all("Score send failed!");
	}
	//write screenshot
	std::cout << "size: " << size << ", name: " << SS_SHM_NAME << ", SIZE: " << SS_SHM_SIZE << std::endl;
	clock_t begin = clock();
	ind = shm_send_buffer_no_sem(buffer, size, SS_SHM_NAME, SS_SHM_SIZE);
	clock_t end = clock();
	std::cout << "Written screenshot in: " << (double)(end - begin) / CLOCKS_PER_SEC << " seconds." << std::endl;
	if(ind < 0)
	{
		close_all("Screenshot send failed!");
	}
	// semaphore close
	sem_post(sem);
	sem_close(sem);
	std::cout << "Game info successfully sent!" << std::endl;
	return 0;
}
