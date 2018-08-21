//FFTW library 
#include <fftw3.h>
//Shared Memory 
#include "CSharedMemSimple.hpp"
#include "ShMemSymBuff_cucomplex.hpp"
#include "cpuLS.hpp"
#include <csignal>
#include <fstream>
#define mode 0

/*
	mode:
		= 1 -> master -> creates shared memory 
		= 0 -> slave -> doesn't create the shared memory
		
	Waits to read dimension vector then does fft on it and then divides by 1+i 
*/

//! Install dependencies: apt-get -y install libboost-program-options-dev libfftw3-dev 
//!How to Compile:   g++ -o cpu ../../examples/cpuLS.cpp -lfftw3f -lrt
// ./cpu

//LS
//Y = 16 x 1024
//X = 1 x 1023
//H = 16 x 1023
using namespace std;
static bool stop_signal_called = false;
void sig_int_handler(int){stop_signal_called = true;}

int main(){
	int rows = numOfRows; // number of vectors -> 16
	int cols = dimension;//dimension -> 1024
	string shm_uid = shmemID;
	
	//printf("CPU LS: \n");
	//printInfo();
	
	//Y = 16x1024
	complexF* Y = 0;
	Y = (complexF*)malloc(rows*cols*sizeof(*Y));
	fftOneRow(Y, cols, 0);
	//H (and Hconj) = 16x1023
	complexF* Hconj = 0;
	Hconj = (complexF *)malloc(rows*(cols-1)* sizeof (*Hconj));
	//X = 1x1023 -> later can become |H|^2
	complexF* X = 0;
	X = (complexF *)malloc((cols-1)* sizeof(*X));
	
	// Create shared memory space, return pointer and set as master.
	buffPtr=new ShMemSymBuff(shm_uid, mode);
	std::signal(SIGINT, &sig_int_handler);
	
	//Find H* (H conjugate) ->16x1023 and |H|^2 -> 1x1023
	std:;string file = "ChanEst.dat";
	for (int iter = 0; iter < numberOfSymbolsToTest; iter++) {
		firstVector(Y, Hconj, X, rows, cols, iter);
		if (iter == 0) {
			outfile.open(file.c_str(), std::ofstream::binary | std::ofstream::trunc);
		} else {
			outfile.open(file.c_str(), std::ofstream::binary | std::ofstream::app);
		}
		outfile.write((const char*)Hconj, (cols-1)*sizeof(*Hconj));
		outfile.close();
	}
	
	free(Y);
	free(Hconj);
	free(X);
	//delete buffPtr;
	if(timerEn) {
		printTimes(true);
		storeTimes(true);
	}
	
	return 0;

}
