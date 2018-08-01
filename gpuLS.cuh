#ifndef _GPULS_HPP_
#define _GPULS_HPP_

#ifndef cudaEn
	#define cudaEn
#endif

//Shared Memory 
#include "ShMemSymBuff_cucomplex.hpp"
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <assert.h>

#define FFT_size dimension
#define cp_size prefix
#define numSymbols lenOfBuffer

//gpu

#define threadsPerBlock FFT_size
#define numOfBlocks numOfRows

//LS
#define fileNameForX "Pilots.dat"
#define mode 0
/*
	mode:
		= 1 -> master -> creates shared memory 
		= 0 -> slave -> doesn't create the shared memory
*/
 
//!How to Compile:   nvcc ../../examples/gpuLS_cucomplex.cu -lcufft -lrt -o gpu -arch=sm_35
// ./gpu

//LS
//Y = 16 x 1024
//X = 1 x 1023
//H = 16 x 1023
ShMemSymBuff* buffPtr;
using namespace std;
	
	
//Reads in Vector X from file -> 1xcols
void matrix_readX(cuFloatComplex* X, int cols){
	ifstream inFile;
	inFile.open(fileNameForX);
	if (!inFile) {
		cerr << "Unable to open file "<< fileNameForX<<", filling in 1+i for x\n";
		float c=1.0f;
		for (int col = 0; col <  cols; col++){
			X[col].x = c;
			X[col].y = c;
		}
		return;
	}
	inFile.read((char*)X, (cols)*sizeof(*X));
	/*
	float c=0;
	for (int col = 0; col <  cols; col++){
		inFile >> c;
		X[col].real=c;
		inFile >> c;
		X[col].imag=c;
	}
	*/
	cuFloatComplex* temp = 0;
	temp=(cuFloatComplex*)malloc ((cols-1)/2* sizeof (*temp));
	//copy second half to temp
	memmove(temp, &X[(cols+1)/2], (cols-1)/2* sizeof (*X));
	//copy first half to second half
	memmove(&X[(cols-1)/2], X, (cols+1)/2* sizeof (*X));
	//copy temp to first half
	memmove(X, temp, (cols-1)/2* sizeof (*X));
	
	free(temp);
	inFile.close();
}

void copyPilotToGPU(cuFloatComplex* dX, int rows, int cols) {
	//X = 1x1023 -> later can become |H|^2
	cuFloatComplex* X = 0;
	int sizeX=rows*(cols-1)* sizeof(*X);
	X = (cuFloatComplex*)malloc(sizeX);
	//cuFloatComplex* H =0;
	//H = (cuFloatComplex *)malloc(sizeX*rows);
	//cudaMalloc((void**)&H, size);
	
	//Read in X vector -> 1x1023
	for (int i = 0; i < rows; i++) {
		//std::cout << "Here...\n";
		matrix_readX(&X[i*(cols-1)], cols-1);
	}
	//std::cout << "Here...\n";
	cudaMemcpy(dX, X, rows*(cols-1)*sizeof(*dX), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}


__global__ void shiftOneRow(cuFloatComplex* Y, int cols, int rows){
	int col = threadIdx.x;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	extern __shared__ cuFloatComplex temp[];
	
	if (threadIdx.x < (cols+1)/2) {
		temp[col] = Y[tid+((cols-1)/2)];
	} else if (threadIdx.x >= (cols+1)/2) {
		temp[col] = Y[tid-((cols+1)/2)];
	}
	__syncthreads();
	
	Y[tid] = temp[col];
	__syncthreads();
}



void shiftOneRowCPU(cuFloatComplex* Y, int cols, int row){
	cuFloatComplex* Yf = &Y[row*cols];
	//std::cout << "Here...\n";
	cuFloatComplex* temp = 0;
	temp=(cuFloatComplex*)malloc ((cols+1)/2* sizeof (*temp));
	//copy second half to temp
	memmove(temp, &Yf[(cols-1)/2], (cols+1)/2* sizeof (*Yf));
	//copy first half to second half
	memmove(&Yf[(cols+1)/2], Yf, (cols-1)/2* sizeof (*Yf));
	//copy temp to first half
	memmove(Yf, temp, (cols+1)/2* sizeof (*Yf));
	
	free(temp);
	
}

__global__ void dropPrefix(cuFloatComplex *Y, cuFloatComplex *dY, int rows1, int cols1){
	
	int rows = rows1;
	int cols= cols1;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	Y[tid] = dY[blockIdx.x*(blockDim.x+prefix) + threadIdx.x + prefix];
	/*
	for(int i =0; i<rows; i++){
		memcpy(&Y[i*cols], &dY[i*(cols+prefix)+ prefix], cols*sizeof(*dY));
	}	
	*/
	
	
}

__global__ void findHs(cuFloatComplex* dY,cuFloatComplex* dH,cuFloatComplex* dX, int rows1, int cols1){	
	//int cols=cols1-1;
	//find my work
	//Drop first element and copy it into Hconj
	dH[blockIdx.x*blockDim.x + threadIdx.x] = dY[blockIdx.x*(blockDim.x + 1) + threadIdx.x + 1];
	__syncthreads();
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	//complex division
	//H/X where H = FFT(Y) (w/ dropped first element)
	//Then take conjugate of H
	if (threadIdx.x < blockDim.x) {
		dH[blockIdx.x*blockDim.x + threadIdx.x] = cuCdivf(dH[blockIdx.x*blockDim.x + threadIdx.x], dX[blockIdx.x*blockDim.x + threadIdx.x]);
		dH[blockIdx.x*blockDim.x + threadIdx.x] = cuConjf(dH[blockIdx.x*blockDim.x + threadIdx.x]);
		dX[blockIdx.x*blockDim.x + threadIdx.x].x = dH[blockIdx.x*blockDim.x + threadIdx.x].x * dH[blockIdx.x*blockDim.x + threadIdx.x].x + dH[blockIdx.x*blockDim.x + threadIdx.x].y * dH[blockIdx.x*blockDim.x + threadIdx.x].y;
	}
	__syncthreads();
	//Now dH holds conj H
	
	
}


__global__ void findDistSqrd(cuFloatComplex* H, float* Hsqrd, int rows, int cols){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	extern __shared__ cuFloatComplex temp[];
	int sid = threadIdx.x*cols + blockIdx.x;
	temp[threadIdx.x] = H[sid];
	
	for (int i = 1; i < rows; i = i*2) {
		if (threadIdx.x%(2*i) == 0) {
			temp[threadIdx.x].x += temp[threadIdx.x+i].x;
		}
		__syncthreads();
	}
	
	
	if(threadIdx.x == 0) {
		Hsqrd[blockIdx.x] = temp[threadIdx.x].x;
	}
}


void firstVector(cuFloatComplex* dY, cuFloatComplex* Y, cuFloatComplex* dH, cuFloatComplex* dX, float* Hsqrd, int rows, int cols, int it){
	clock_t start, finish;
	//std::cout << "Here...\n";	
	
	// CUFFT plan -> do it one time before?
	
	
	//Read in Y with prefix
	buffPtr->readNextSymbol(dY, it);
	
	if(timerEn){
		start = clock();
	}
	cudaMemcpy(Y, dY, rows*cols*sizeof(*Y), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	if(timerEn){
		finish = clock();
		readT[it] = readT[it] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	if(timerEn){
		start = clock();
	}
	
	
	//FFT(Y)
	cufftHandle plan;
	cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	if(timerEn){
		finish = clock();
		fft[it] = fft[it]+ ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	//find Hconj and Hsqrd
	if(timerEn){
		start = clock();
	}
//	dim3 dimBlock(numOfBlocks, threadsPerBlock-1);
	findHs<< <numOfBlocks, threadsPerBlock-1>> >(Y, dH, dX, rows, cols);
	cudaDeviceSynchronize();
	//Save |H|^2 into Hsqrd
	findDistSqrd<< <threadsPerBlock-1, numOfBlocks, numOfBlocks*sizeof(cuFloatComplex)>> >(dX, Hsqrd, rows, cols-1);
	cudaDeviceSynchronize();
	
	if(timerEn){
		finish = clock();
		decode[it] = decode[it] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	//free(X);
	//cudaFree(H);
	
	//dH holds H conj
	//dX holds {H^2)	
}


__global__ void multiplyWithChannelConj(cuFloatComplex* Y, cuFloatComplex* Hconj, cuFloatComplex* Yf, int rows1, int cols1){
	int rows = rows1;
	int cols= cols1;
    
    //find my work 
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1023
	int row = blockIdx.x;
	int sym = blockIdx.y;
	int j = threadIdx.x;
	Yf[sym*blockDim.x*gridDim.x + row*blockDim.x + j] = Y[sym*(blockDim.x+1)*gridDim.x + row*(blockDim.x+1) + j + 1];
	__syncthreads();
	
	if (j < cols-1) {
		Yf[sym*blockDim.x*gridDim.x + row*blockDim.x + j] = cuCmulf(Yf[sym*blockDim.x*gridDim.x + row*blockDim.x + j],Hconj[row*blockDim.x + j]);
	}
	__syncthreads();
}


__global__ void combineForMRC(cuFloatComplex* Y, float* Hsqrd, int rows1, int cols1) {
	int rows = rows1;
	int cols = cols1;
	int row = blockIdx.x;
	int col = threadIdx.x;
	//int tid = blockIdx.x*blockDim.x + threadIdx.x;
	extern __shared__ cuFloatComplex temp[];
	int sid = threadIdx.x*cols + blockIdx.x + rows*cols*blockIdx.y;
	temp[col] = Y[sid];
	
	for (int i = 1; i < rows; i = i*2) {
		if (threadIdx.x%(2*i) == 0) {
			temp[col] = cuCaddf(temp[col],temp[col+i]);
		}
		__syncthreads();
	}
	
	if (col == 0) {
		Y[row + cols*blockIdx.y].x = temp[col].x/Hsqrd[row];
		Y[row + cols*blockIdx.y].y = temp[col].y/Hsqrd[row];
		__syncthreads();
	}
}


void demodOneSymbol(cuFloatComplex *dY, cuFloatComplex* Y, cuFloatComplex *Hconj, float *Hsqrd,int rows1, int cols1, int it) {
	int rows = rows1;
	int cols= cols1;

	clock_t start, finish;
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1024
	
	if(it==numberOfSymbolsToTest-1){
		//if last one
		buffPtr->readLastSymbol(dY);
	} else {
		buffPtr->readNextSymbol(dY, it);
	}
	
	if(timerEn){
		start = clock();
	}
	cudaMemcpy(Y, dY, rows*cols*sizeof(*Y), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	if(timerEn){
		finish = clock();
		readT[it] = readT[it] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	if(timerEn){
		start = clock();
	}
	
	//FFT(Y)
	cufftHandle plan;
	cufftPlan1d(&plan, threadsPerBlock, CUFFT_C2C, numOfBlocks);
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	if(timerEn){
		finish = clock();
		fft[it] = fft[it]+ ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	cuFloatComplex* Yf = 0;
	cudaMalloc((void**)&Yf, rows*(cols-1)* sizeof (*Yf));
	
	if(timerEn){
		start = clock();
	}
	multiplyWithChannelConj<< <numOfBlocks, threadsPerBlock-1>> >(Y, Hconj, Yf, rows, cols);
	cudaDeviceSynchronize();
	combineForMRC<< <threadsPerBlock-1, numOfBlocks, numOfBlocks*sizeof(cuFloatComplex)>> >(Yf, Hsqrd, rows, cols-1);
	cudaDeviceSynchronize();
	cudaMemcpy(dY, Yf, (cols-1)*sizeof(*dY), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	shiftOneRowCPU(dY,cols-1,0);
	
	if(timerEn){
		finish = clock();
		decode[it] = decode[it] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	cudaFree(Yf);
	cudaDeviceSynchronize();
}

void demodOneFrame(cuFloatComplex *dY, cuFloatComplex* Y, cuFloatComplex* dX, cuFloatComplex *Hconj, float *Hsqrd, int rows1, int cols1) {
	int rows = rows1;
	int cols = cols1;

	clock_t start, finish;
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1024
	
	for (int it = 0; it < numberOfSymbolsToTest; it++) {
		if(it==numberOfSymbolsToTest-1){
			//if last one
			buffPtr->readLastSymbol(&dY[rows*cols*it]);
		} else {
			buffPtr->readNextSymbol(&dY[rows*cols*it], it);
		}
	}
	
	if(timerEn){
		start = clock();
	}
	cudaMemcpy(Y, dY, rows*cols*(lenOfBuffer)*sizeof(*Y), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	if(timerEn){
		finish = clock();
		readT[1] = readT[1] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	if(timerEn){
		start = clock();
	}
	
	//FFT(Y)
	cufftHandle plan;
	cufftPlan1d(&plan, cols, CUFFT_C2C, rows*(lenOfBuffer));
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	if(timerEn){
		finish = clock();
		fft[1] = fft[1]+ ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	
	if(timerEn){
		start = clock();
	}
//	dim3 dimBlock(numOfBlocks, threadsPerBlock-1);
	findHs<< <numOfBlocks, threadsPerBlock-1>> >(Y, Hconj, dX, rows, cols);
	cudaDeviceSynchronize();
	//Save |H|^2 into Hsqrd
	findDistSqrd<< <threadsPerBlock-1, numOfBlocks, numOfBlocks*sizeof(cuFloatComplex)>> >(dX, Hsqrd, rows, cols-1);
	cudaDeviceSynchronize();
	
	if(timerEn){
		finish = clock();
		decode[0] = decode[0] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	if(timerEn){
		start = clock();
	}
	cuFloatComplex* Yf = 0;
	cudaMalloc((void**)&Yf, rows*(cols-1)*(lenOfBuffer-1)* sizeof (*Yf));
	dim3 gridDims1(numOfBlocks, lenOfBuffer-1);
	multiplyWithChannelConj<< <gridDims1, threadsPerBlock-1>> >(&Y[rows*cols], Hconj, Yf, rows, cols);
	cudaDeviceSynchronize();
	dim3 gridDims2(threadsPerBlock-1, lenOfBuffer-1);
	combineForMRC<< <gridDims2, numOfBlocks, numOfBlocks*sizeof(cuFloatComplex)>> >(Yf, Hsqrd, rows, cols-1);
	cudaDeviceSynchronize();
	shiftOneRow<< <lenOfBuffer-1, threadsPerBlock-1, (threadsPerBlock-1)*sizeof(cuFloatComplex)>> >(Yf, cols-1, rows);
	cudaDeviceSynchronize();
	cudaMemcpy(dY, Yf, (cols-1)*(lenOfBuffer-1)*sizeof(*dY), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	/*
	for (int it = 1; it < lenOfBuffer-1; it++) {
		shiftOneRowCPU(dY,cols-1,0);
	}
	*/
	if(timerEn){
		finish = clock();
		decode[1] = decode[1] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	cudaFree(Yf);
	cudaDeviceSynchronize();
}

#endif