#ifndef cudaEn
	#define cudaEn
#endif

//Shared Memory 
#include "ShMemSymBuff_gpu.hpp"
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
//#include <boost/thread.hpp>
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
	cudaGetDeviceProperties(&gpu->devProp, device_number);
	int maxThreads = gpu->devProp.maxThreadsPerBlock;
	
	//dY holds symbol with prefix
	cuFloatComplex *dY = 0;
	cudaMalloc((void**)&dY, rows*(cols-1)*lenOfBuffer* sizeof (*dY));
	
	float *Hsqrd = 0;
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
	
	float frameTime;
	
	cufftHandle plan;
	cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	
	gpuLS *gpu = new gpuLS;
	
	//Shared Memory
	string shm_uid = shmemID;
	buffPtr=new ShMemSymBuff(shm_uid, mode);
	std::signal(SIGINT, &sig_int_handler);
	
	gpu->copyPilotToGPU(dX, rows, cols);
	
	
	cudaStream_t stream[lenOfBuffer];
	dim3 gridDim(rows,0,0);
	dim3 blockDim(cols,0,0);
	clock_t start, finish;
	clock_t start_total;
	while (not stop_signal_called) {
		start_total = clock();
		&stream[0] = gpu->buffPtr->createStream(0);
		gpu->buffPtr->readNextSymbolCUDA(Y, 0);
		if(timerEn){
			start = clock();
		}
		gpu->batchedFFT(Y, rows, cols, stream[0]);
		if(timerEn){
			finish = clock();
			buffPtr->setFft(((float)(finish - start))/(float)CLOCKS_PER_SEC, 0);
		}
		
		
		if(timerEn){
			start = clock();
		}
		gpu->FindLeastSquaresGPU(, dH, dX, rows, cols, blockDim, gridDim, &stream[0]);
		gpu->FindHsqrdforMRC(dH, Hsqrd, rows, cols, blockDim, gridDim, &stream[0]);
		if(timerEn){
			finish = clock();
			buffPtr->setDecode(((float)(finish - start))/(float)CLOCKS_PER_SEC, 0);
		}
		
		for (int iter = 1; iter < numSymbols; iter++) {
			&stream[iter] = gpu->buffPtr->createStream(iter);
			if(it==numberOfSymbolsToTest-1){
				//if last one
				gpu->buffPtr->readLastSymbolCUDA(&Y[rows*cols*iter]);
			} else {
				gpu->buffPtr->readNextSymbolCUDA(&Y[rows*cols*iter], iter);
			}
			
			if(timerEn){
				start = clock();
			}
			gpu->batchedFFT(&Y[rows*cols*iter], rows, cols, &stream[0]);
			if(timerEn){
				finish = clock();
				buffPtr->setFft(((float)(finish - start))/(float)CLOCKS_PER_SEC, iter);
			}
			
			cudaStreamSynchronize(stream[0]);
			
			if(timerEn){
				start = clock();
			}
			gpu->MultiplyWithChannelConj(&Y[rows*cols*iter], dH, &dY[rows*(cols-1)*iter], rows, cols, 1, blockDim, gridDim, &stream[iter]);
			gpu->CombineForMRC(&dY[rows*(cols-1)*iter], Hsqrd, rows, cols, blockDim, gridDim, &stream[iter]);
			gpu->ShiftOneRow(&dY[rows*(cols-1)*iter], cols, 1, blockDim, gridDim, &stream[iter]);
			if(timerEn){
				finish = clock();
				buffPtr->setDecode(((float)(finish - start))/(float)CLOCKS_PER_SEC, iter);
			}
			
			if(testEn){
				//printf("Symbol #%d:\n", i);
				//cuda copy it over
				cudaMemcpyAsync(&Yf[(cols-1)*(iter-1)], &dY[rows*(cols-1)*iter], (cols-1)* sizeof (*Yf), cudaMemcpuDeviceToHost, stream[iter]);
				outfile.open(file.c_str(), std::ofstream::binary | std::ofstream::trunc);
				outfile.write((const char*)&Yf[(cols-1)*(iter-1)], (cols-1)*sizeof(*Yf));
				outfile.close();
			}
			gpu->buffPtr->destroyStream(iter);
		}
		gpu->buffPtr->destroyStream(0);
		if(timerEn) {
			gpu->buffPtr->printTimes(true);
			gpu->buffPtr->storeTimes(false);
		}
		
		//Resetting the device and allocating memory for next frame
		cudaDeviceReset();
		cudaMalloc((void**)&Hsqrd, (cols-1)* sizeof (*Hsqrd));
		cudaMalloc((void**)&dH, rows*(cols-1)* sizeof (*dH));
		cudaMalloc((void**)&dX, rows*(cols-1)* sizeof (*dX));
		cudaMalloc((void**)&Y, rows*cols*lenOfBuffer*sizeof(*Y));
		cudaMalloc((void**)&dY, rows*(cols-1)*lenOfBuffer* sizeof (*dY));
		cufftHandle plan;
		cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
		cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
		cufftDestroy(plan);
		while ((((float)(clock() - start_total))/(float)CLOCKS_PER_SEC) < 1);
	}
	
	free(Yf);
	cudaFree(dY);
	cudaFree(Y);
	cudaFree(dH);
	cudaFree(dX);
	cudaFree(Hsqrd);
	delete(gpu);
	cudaDeviceReset();
	return 0;

}