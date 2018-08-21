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

//! Packages needed to run this program - Openblas FFTW lapack lgfortran
//! Install dependencies: apt-get -y install libboost-program-options-dev libfftw3-dev -libopenblas-dev
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
	int users = numUsers;
	string shm_uid = shmemID;
	
	//printf("CPU LS: \n");
	//printInfo();
	
	//Y = 16x1024
	complexF* Y = 0;
	Y = (complexF *)calloc(rows*(cols+prefix)*numberOfSymbolsToTest, sizeof(*Y));
	for (int i = 0; i < rows; i++) {
		ifftOneRow(Y, cols, i);
	}
	complexF* dY = 0;
	dY = (complexF *)calloc(rows*(cols+prefix), sizeof(*dY));
	//H (and Hconj) = 16x1023
	complexF* Hconj = 0;
	Hconj = (complexF *)calloc(rows*cols*users, sizeof (*Hconj));
	//X = 1x1023 -> later can become |H|^2
	complexF* X = 0;
	X = (complexF *)calloc(rows*(cols-1)*users*numberOfSymbolsToTest, sizeof(*X));
	
	complexF* dX = 0;
	dX = (complexF *)calloc(rows*(cols-1)*users, sizeof(*X));
	
	// Create shared memory space, return pointer and set as master.
	//buffPtr=new ShMemSymBuff(shm_uid, mode);
	std::signal(SIGINT, &sig_int_handler);

	std::ifstream infile;
	clock_t start, finish;
	modRefSymbol(Y, X, cols);
	outfile.open("refSymbol.dat", std::ofstream::binary | std::ofstream::trunc);
	for (int i = 0; i < rows; i++) {
		outfile.write((const char*)Y, (cols+prefix)*sizeof(*Y));
	}
	outfile.close();
	
	free(Y);
	free(dY);
	free(Hconj);
	free(X);
	free(dX);
	//delete buffPtr;
	
	if(timerEn) {
		printTimes(true);
		//storeTimes(true);
	}
	
	return 0;

}
