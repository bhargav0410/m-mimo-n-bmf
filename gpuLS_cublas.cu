#ifndef cudaEn
	#define cudaEn
#endif

//Shared Memory 
#include "ShMemSymBuff_cucomplex.hpp"
#include "gpuLS.cuh"
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <assert.h>
#include <csignal>
#define FFT_size dimension
#define cp_size prefix
#define numSymbols lenOfBuffer


/*
	mode:
		= 1 -> master -> creates shared memory 
		= 0 -> slave -> doesn't create the shared memory
*/
 
//!How to Compile:   nvcc ../../examples/gpuLS_full_frame.cu -lcufft -lrt -o gpu -arch=sm_35
// ./gpu

//LS
//Y = 16 x 1024
//X = 1 x 1023
//H = 16 x 1023
using namespace std;

static bool stop_signal_called = false;
void sig_int_handler(int){stop_signal_called = true;}

std::string file = "Output_gpu.dat";
//std::ofstream outfile;

int main(){
	int rows = numOfRows; // number of vectors
	int cols=dimension;//dimension
	device_number = 0;
	cudaSetDevice(device_number);
	cudaGetDeviceProperties(&devProp, device_number);
	
	//dY holds symbol with prefix
	cuFloatComplex *dY = 0;
	dY = (cuFloatComplex*)malloc(rows*(cols)*lenOfBuffer* sizeof (*dY));
	
	cuFloatComplex *Hsqrd = 0;
	cudaMalloc((void**)&Hsqrd, (cols-1)* sizeof (*Hsqrd));
	
	//dH (and Hconj) = 16x1023
	cuFloatComplex *dH = 0;
	cudaMalloc((void**)&dH, rows*(cols-1)* sizeof (*dH));
	
	//X = 1x1023 -> later can become |H|^2
	cuFloatComplex *dX = 0;
	cudaMalloc((void**)&dX, rows*(cols-1)* sizeof (*dX));
	
	cuFloatComplex *Yf = 0;
	Yf = (cuFloatComplex*)malloc((cols-1)*(lenOfBuffer-1)* sizeof (*Yf));
	
	cuFloatComplex* Y = 0;
	cudaMalloc((void**)&Y, rows*cols*lenOfBuffer*sizeof(*Y));
	
	clock_t start, finish;
	float frameTime;
	
	cufftHandle plan;
	cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	
	
	//Shared Memory
	string shm_uid = shmemID;
	buffPtr=new ShMemSymBuff(shm_uid, mode);
	std::signal(SIGINT, &sig_int_handler);
	
	copyPilotToGPU(dX, rows, cols);
		
	while (not stop_signal_called) {
		start = clock();
		demodOneFrameCUDA(dY, Y, dX, dH, Hsqrd, rows, cols);
		if (timerEn) {
			printTimes(true);
			storeTimes(false);
		}
		if(testEn){
			//printf("Symbol #%d:\n", i);
			//cuda copy it over
			memcpy(Yf, dY, (cols-1)*(lenOfBuffer-1)* sizeof (*Yf));
			outfile.open(file.c_str(), std::ofstream::binary | std::ofstream::trunc);
			outfile.write((const char*)Yf, (cols-1)*(lenOfBuffer-1)*sizeof(*Yf));
			outfile.close();
		}
		while ((((float)(clock() - start))/(float)CLOCKS_PER_SEC) < 1);
	}
	
	free(Yf);
	free(dY);
	cudaFree(Y);
	cudaFree(dH);
	cudaFree(dX);
	cudaFree(Hsqrd);
	
	if(timerEn) {
		printTimes(true);
		storeTimes(false);
	}
	
	return 0;

}