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
	Y = (complexF*)malloc(rows*(cols+prefix)*sizeof(*Y));
	//H (and Hconj) = 16x1023
	complexF* Hconj = 0;
	Hconj = (complexF *)calloc(rows*cols*users, sizeof (*Hconj));
	//X = 1x1023 -> later can become |H|^2
	complexF* X = 0;
	X = (complexF *)calloc(rows*cols*users, sizeof(*X));
	
	// Create shared memory space, return pointer and set as master.
	//buffPtr=new ShMemSymBuff(shm_uid, mode);
	std::signal(SIGINT, &sig_int_handler);
	
	std::string file = "refSymbolTest.dat";
	modRefSymbol(Y, X, rows, cols);
	outfile.open(file.c_str(), std::ofstream::binary);
	outfile.write((const char*)Y, rows*(cols+prefix)*sizeof(*Y));
	outfile.close();
	
	
	
	/*
	for (int i = 0; i < 6; i++) {
		X[i].real = rand()%10;
		X[i].imag = rand()%10;
		std::cout << "( " << X[i].real << ", " << X[i].imag << ")";
	}
	std::cout << "\n";
	
	//X[0].real = 7;X[1].real = 8;X[2].real = 11;X[3].real = 13;X[4].real = 16;X[5].real = 21;
	//Hconj[0].real = 1;Hconj[1].real = 2;Hconj[2].real = 3;Hconj[3].real = 4;
	
	createZeroForcingMatrix(Hconj, X, 3, 2, 2);
	
	multiplyWithChannelInv(Y, Hconj, X, 3, 2, 2);
	
	for (int i = 0; i < 3*3 ; i++) {
		std::cout << "( " << Y[i].real << ", " << Y[i].imag << ")";
	}
	std::cout << "\n";
//	std::cout << "H matrix: " << Hconj[0].real << ", " << Hconj[1].real << ", " << Hconj[2].real << ", " << Hconj[3].real << std::endl;
	
	*/
	
	free(Y);
	free(Hconj);
	free(X);
	//delete buffPtr;
	/*
	if(timerEn) {
		printTimes(true);
		storeTimes(true);
	}
	*/
	return 0;

}
