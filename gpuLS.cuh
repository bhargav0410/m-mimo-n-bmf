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
#include <math.h>

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
int device_number = 0;
cudaDeviceProp devProp;
	
	
//Reads in Vector X from file -> 1xcols
inline void matrix_readX(cuFloatComplex* X, int cols){
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
	free(X);
}


__global__ void shiftOneRow(cuFloatComplex* Y, int cols1, int rows1){
	int cols = cols1;
	//int rows = rows1;
	int col = threadIdx.y*cols + threadIdx.x;
	int tid = blockIdx.y*gridDim.x*blockDim.y*cols + blockIdx.x*blockDim.y*cols + threadIdx.y*cols + threadIdx.x;
	extern __shared__ cuFloatComplex temp[];
	
	if ((threadIdx.x + blockIdx.x*cols) < (cols+1)/2) {
		temp[col] = Y[tid+((cols-1)/2)];
	} else if ((threadIdx.x + blockIdx.x*cols) >= (cols+1)/2 and threadIdx.x < cols) {
		temp[col] = Y[tid-((cols+1)/2)];
	}
	__syncthreads();
	
	Y[tid] = temp[col];
	__syncthreads();
}



inline void shiftOneRowCPU(cuFloatComplex* Y, int cols, int row){
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
	int cols = cols1-1;
	int rows = rows1;
	int tid = (blockIdx.y*gridDim.x*blockDim.y + blockIdx.x*blockDim.y + threadIdx.y)*cols + threadIdx.x;
	int tid2 = (blockIdx.y*gridDim.x*blockDim.y + blockIdx.x*blockDim.y + threadIdx.y)*(cols+1) + threadIdx.x + 1;
	//find my work
	//Drop first element and copy it into Hconj
	if ((blockIdx.y + threadIdx.y)*blockDim.x + threadIdx.x < cols) {
		dH[tid] = dY[tid2];
	}
	__syncthreads();
	
	//complex division
	//H/X where H = FFT(Y) (w/ dropped first element)
	//Then take conjugate of H
	if (tid < cols*rows) {
		dH[tid] = cuCdivf(dH[tid], dX[tid]);
		dH[tid] = cuConjf(dH[tid]);
		//dX[tid].x = dH[tid].x * dH[tid].x + dH[tid].y * dH[tid].y;
	}
	__syncthreads();
	//Now dH holds conj H
	
	
}


__global__ void findDistSqrd(cuFloatComplex* H, float* Hsqrd, int rows1, int cols1){
	int cols = cols1;
	int rows = rows1;
	//int tid = blockIdx.x*cols + threadIdx.x;
	extern __shared__ cuFloatComplex temp[];
	int sid = threadIdx.x*cols + blockIdx.x*blockDim.y + threadIdx.y;
	int tempID = threadIdx.y*rows + threadIdx.x;
	if (sid < rows*cols) {
		temp[tempID] = H[sid];
	}
	temp[tempID].x = temp[tempID].x*temp[tempID].x + temp[tempID].y*temp[tempID].y;
	__syncthreads();
	for (int i = 1; i < rows; i = i*2) {
		if (threadIdx.x%(2*i) == 0 and (blockIdx.x*blockDim.y + threadIdx.y) < cols) {
			temp[tempID].x += temp[tempID+i].x;
		}
		__syncthreads();
	}
	
	
	if(threadIdx.x == 0 and (blockIdx.x*blockDim.y + threadIdx.y) < cols) {
		Hsqrd[blockIdx.x*blockDim.y + threadIdx.y] = temp[tempID].x;
	}
}


__global__ void multiplyWithChannelConj(cuFloatComplex* Y, cuFloatComplex* Hconj, cuFloatComplex* Yf, int rows1, int cols1,int syms1 = 1){
	int rows = rows1;
	int cols = cols1-1;
    int syms = syms1;
    //find my work 
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1023
	int row = blockIdx.x;
	int sym = blockIdx.z;
	int j = threadIdx.x;
	int tid = (blockIdx.z*gridDim.y*gridDim.x*blockDim.y + blockIdx.y*gridDim.x*blockDim.y + blockIdx.x*blockDim.y + threadIdx.y)*cols + threadIdx.x;
	int tid2 = (blockIdx.z*gridDim.y*gridDim.x*blockDim.y + blockIdx.y*gridDim.x*blockDim.y + blockIdx.x*blockDim.y + threadIdx.y)*(cols+1) + threadIdx.x;
	
	if ((blockIdx.y + threadIdx.y)*cols + threadIdx.x < cols) {
		Yf[tid] = Y[tid2];
	}
	__syncthreads();
	
	if (tid < rows*cols*syms) {
		Yf[tid] = cuCmulf(Yf[tid],Hconj[row*blockDim.x + j]);
	}
	__syncthreads();
}


__global__ void combineForMRC(cuFloatComplex* Y, float* Hsqrd, int rows1, int cols1) {
	int rows = rows1;
	int cols = cols1;
	int row = blockIdx.x*blockDim.y + threadIdx.y;
	int col = threadIdx.x;
	//int tid = blockIdx.x*blockDim.x + threadIdx.x;
	extern __shared__ cuFloatComplex temp[];
	int tempID = threadIdx.y*rows + threadIdx.x;
	int sid = blockIdx.y*rows*cols + threadIdx.x*cols + blockIdx.x*blockDim.y + threadIdx.y;
	temp[tempID] = Y[sid];
	
	for (int i = 1; i < rows; i = i*2) {
		if (threadIdx.x%(2*i) == 0 and (blockIdx.x*blockDim.y + threadIdx.y) < cols) {
			temp[tempID] = cuCaddf(temp[tempID],temp[tempID+i]);
		}
		__syncthreads();
	}
	
	if (threadIdx.x == 0 and (blockIdx.x*blockDim.y + threadIdx.y) < cols) {
		Y[row + cols*blockIdx.y].x = temp[tempID].x/Hsqrd[row];
		Y[row + cols*blockIdx.y].y = temp[tempID].y/Hsqrd[row];
		__syncthreads();
	}
}






/*-----------------------------------Host Functions--------------------------------------*/

inline void firstVector(cuFloatComplex* dY, cuFloatComplex* Y, cuFloatComplex* dH, cuFloatComplex* dX, float* Hsqrd, int rows, int cols, int it){
	clock_t start, finish;
	//std::cout << "Here...\n";	
	
	// CUFFT plan -> do it one time before?
	
	
	//Read in Y with prefix
	buffPtr->readNextSymbolCUDA(dY, it);
	
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
	findDistSqrd<< <threadsPerBlock-1, numOfBlocks, numOfBlocks*sizeof(cuFloatComplex)>> >(dH, Hsqrd, rows, cols-1);
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

inline void demodOneSymbol(cuFloatComplex *dY, cuFloatComplex* Y, cuFloatComplex *Hconj, float *Hsqrd,int rows1, int cols1, int it) {
	int rows = rows1;
	int cols= cols1;

	clock_t start, finish;
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1024
	
	if(it==numberOfSymbolsToTest-1){
		//if last one
		buffPtr->readLastSymbolCUDA(dY);
	} else {
		buffPtr->readNextSymbolCUDA(dY, it);
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

inline void demodOneFrame(cuFloatComplex *dY, cuFloatComplex* Y, cuFloatComplex* dX, cuFloatComplex *Hconj, float *Hsqrd, int rows1, int cols1) {
	int rows = rows1;
	int cols = cols1;
	int maxThreads = devProp.maxThreadsPerBlock;
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
	if (threadsPerBlock <= maxThreads) {
		findHs<< <numOfBlocks, threadsPerBlock-1>> >(Y, Hconj, dX, rows, cols);
		cudaDeviceSynchronize();
		//Save |H|^2 into Hsqrd
	} else {
		dim3 chanEstDim(numOfBlocks,ceil(threadsPerBlock/maxThreads));
		findHs<< <chanEstDim, maxThreads>> >(Y, Hconj, dX, rows, cols);
		cudaDeviceSynchronize();
		//Save |H|^2 into Hsqrd
	}
	findDistSqrd<< <threadsPerBlock-1, numOfBlocks, numOfBlocks*sizeof(cuFloatComplex)>> >(Hconj, Hsqrd, rows, cols-1);
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
	if (threadsPerBlock <= maxThreads) {
		dim3 gridDims1(numOfBlocks, 1, lenOfBuffer-1);
		multiplyWithChannelConj<< <gridDims1, threadsPerBlock-1>> >(&Y[rows*cols], Hconj, Yf, rows, cols, numberOfSymbolsToTest-1);
		cudaDeviceSynchronize();
	} else {
		dim3 gridDims1(numOfBlocks, ceil(threadsPerBlock/maxThreads), lenOfBuffer-1);
		multiplyWithChannelConj<< <gridDims1, maxThreads>> >(&Y[rows*cols], Hconj, Yf, rows, cols, numberOfSymbolsToTest-1);
		cudaDeviceSynchronize();
	}
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

inline void demodOneFrameCUDA(cuFloatComplex* dY, cuFloatComplex* Y, cuFloatComplex* dX, cuFloatComplex *Hconj, float *Hsqrd, int rows1, int cols1) {
	int rows = rows1;
	int cols = cols1;
	int maxThreads = devProp.maxThreadsPerBlock;
	
	clock_t start, finish;
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1024
	/*
	for (int it = 0; it < numberOfSymbolsToTest; it++) {
		if(it==numberOfSymbolsToTest-1){
			//if last one
			buffPtr->readLastSymbolCUDA(&Y[rows*cols*it]);
		} else {
			buffPtr->readNextSymbolCUDA(&Y[rows*cols*it], it);
		}
	}
//	cudaDeviceSynchronize();
	*/
	if(timerEn){
		start = clock();
	}
	
	//FFT(Y)
	cufftHandle plan;
	cufftPlan1d(&plan, cols, CUFFT_C2C, rows*(lenOfBuffer));
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
//	cudaDeviceSynchronize();
	if(timerEn){
		finish = clock();
		fft[1] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	
	if(timerEn){
		start = clock();
	}
//	dim3 dimBlock(numOfBlocks, threadsPerBlock-1);

	if (threadsPerBlock <= maxThreads) {
		findHs<< <numOfBlocks, threadsPerBlock>> >(Y, Hconj, dX, rows, cols);
	//	cudaDeviceSynchronize();
		//Save |H|^2 into Hsqrd
	} else {
		dim3 chanEstBlockDim1(maxThreads);
		dim3 chanEstGridDim1(numOfBlocks,ceil((float)threadsPerBlock/(float)maxThreads));
		findHs<< <chanEstGridDim1, chanEstBlockDim1>> >(Y, Hconj, dX, rows, cols);
	//	cudaDeviceSynchronize();
		//Save |H|^2 into Hsqrd
	}
//	dim3 chanEstBlockDim2(rows,ceil((float)maxThreads/(float)rows));
//	dim3 chanEstGridDim2(ceil((float)(cols)/(ceil((float)maxThreads/(float)rows))),ceil((float)rows/(float)maxThreads));
	findDistSqrd<< <threadsPerBlock-1,numOfBlocks, numOfBlocks*sizeof(cuFloatComplex)>> >(Hconj, Hsqrd, rows, cols-1);
//	cudaDeviceSynchronize();
	
	if(timerEn){
		finish = clock();
		decode[0] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	if(timerEn){
		start = clock();
	}
	cuFloatComplex* Yf = 0;
	cudaMalloc((void**)&Yf, rows*(cols-1)*(lenOfBuffer-1)* sizeof (*Yf));
	if (threadsPerBlock <= maxThreads) {
		dim3 gridDims1(numOfBlocks, 1, lenOfBuffer-1);
		multiplyWithChannelConj<< <gridDims1, threadsPerBlock-1>> >(&Y[rows*cols], Hconj, Yf, rows, cols, numberOfSymbolsToTest-1);
	//	cudaDeviceSynchronize();
	} else {
		dim3 gridDims1(numOfBlocks, ceil((float)threadsPerBlock/(float)maxThreads), lenOfBuffer-1);
		multiplyWithChannelConj<< <gridDims1, maxThreads>> >(&Y[rows*cols], Hconj, Yf, rows, cols, numberOfSymbolsToTest-1);
	//	cudaDeviceSynchronize();
	}
	dim3 gridDims2(threadsPerBlock-1, lenOfBuffer-1);
	combineForMRC<< <gridDims2, numOfBlocks, numOfBlocks*sizeof(cuFloatComplex)>> >(Yf, Hsqrd, rows, cols-1);
	cudaDeviceSynchronize();
	if (threadsPerBlock <= maxThreads) {
		shiftOneRow<< <(1,lenOfBuffer-1), threadsPerBlock-1, (threadsPerBlock-1)*sizeof(cuFloatComplex)>> >(Yf, cols-1, rows);
	//	cudaDeviceSynchronize();
	} else {
		dim3 gridDims3(ceil((float)threadsPerBlock/(float)maxThreads), lenOfBuffer-1);
		shiftOneRow<< <gridDims3, maxThreads, (maxThreads)*sizeof(cuFloatComplex)>> >(Yf, cols-1, rows);
	//	cudaDeviceSynchronize();
	}
	cudaMemcpy(dY, Yf, (cols-1)*(lenOfBuffer-1)*sizeof(*dY), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	/*
	for (int it = 1; it < lenOfBuffer-1; it++) {
		shiftOneRowCPU(dY,cols-1,0);
	}
	*/
	if(timerEn){
		finish = clock();
		decode[1] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	cudaFree(Yf);
}

inline void demodOptimized(cuFloatComplex* dY, cuFloatComplex* Y, cuFloatComplex* dX, cuFloatComplex *Hconj, float *Hsqrd, int rows1, int cols1) {
//	cublasHandle_t handle;
//	cublasCreate(&handle);
	
	int rows = rows1;
	int cols = cols1;
	int maxThreads = devProp.maxThreadsPerBlock;
	
	clock_t start, finish;
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1024
//	cudaDeviceSynchronize();
	
	if(timerEn){
		start = clock();
	}
	
	//FFT(Y)
	cufftHandle plan;
	cufftPlan1d(&plan, cols, CUFFT_C2C, rows*(lenOfBuffer));
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
//	cudaDeviceSynchronize();
	if(timerEn){
		finish = clock();
		fft[1] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	
	if(timerEn){
		start = clock();
	}
//	dim3 dimBlock(numOfBlocks, threadsPerBlock-1);
	if (threadsPerBlock <= maxThreads) {
		dim3 chanEstBlockDim(threadsPerBlock,ceil((float)maxThreads/(float)threadsPerBlock));
		dim3 chanEstGridDim(ceil((float)numOfBlocks/ceil((float)maxThreads/(float)threadsPerBlock)),1);
		findHs<< <numOfBlocks, threadsPerBlock>> >(Y, Hconj, dX, rows, cols);
	//	cudaDeviceSynchronize();
		//Save |H|^2 into Hsqrd
	} else {
		dim3 chanEstBlockDim1(maxThreads);
		dim3 chanEstGridDim1(numOfBlocks,ceil((float)threadsPerBlock/(float)maxThreads));
		findHs<< <chanEstGridDim1, chanEstBlockDim1>> >(Y, Hconj, dX, rows, cols);
	//	cudaDeviceSynchronize();
		//Save |H|^2 into Hsqrd
	}
	
	dim3 distSqrdBlockDim(numOfBlocks,ceil((float)maxThreads/(float)numOfBlocks));
	dim3 distSqrdGridDim(ceil((float)(threadsPerBlock-1)/ceil((float)maxThreads/(float)numOfBlocks)),1);
	findDistSqrd<< <threadsPerBlock-1,numOfBlocks, maxThreads*sizeof(cuFloatComplex)>> >(Hconj, Hsqrd, rows, cols-1);
	
	if(timerEn){
		finish = clock();
		decode[0] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	if(timerEn){
		start = clock();
	}
	cuFloatComplex* Yf = 0;
	cudaMalloc((void**)&Yf, rows*(cols-1)*(lenOfBuffer-1)* sizeof (*Yf));
	if (threadsPerBlock <= maxThreads) {
		dim3 blockDims1(threadsPerBlock,ceil((float)maxThreads/(float)threadsPerBlock));
		dim3 gridDims1(ceil((float)numOfBlocks/ceil((float)maxThreads/(float)threadsPerBlock)), 1, lenOfBuffer-1);
		multiplyWithChannelConj<< <gridDims1, blockDims1>> >(&Y[rows*cols], Hconj, Yf, rows, cols, numberOfSymbolsToTest-1);
	//	cudaDeviceSynchronize();
	} else {
		dim3 gridDims1(numOfBlocks, ceil((float)threadsPerBlock/(float)maxThreads), lenOfBuffer-1);
		multiplyWithChannelConj<< <gridDims1, maxThreads>> >(&Y[rows*cols], Hconj, Yf, rows, cols, numberOfSymbolsToTest-1);
	//	cudaDeviceSynchronize();
	}
	
	dim3 blockDim2(numOfBlocks,ceil((float)maxThreads/(float)numOfBlocks));
	dim3 gridDims2(ceil((float)(threadsPerBlock-1)/ceil((float)maxThreads/(float)numOfBlocks)), lenOfBuffer-1);
	combineForMRC<< <gridDims2, blockDims2, maxThreads*sizeof(cuFloatComplex)>> >(Yf, Hsqrd, rows, cols-1);
	cudaDeviceSynchronize();
	if (threadsPerBlock <= maxThreads) {
		shiftOneRow<< <(1,lenOfBuffer-1), ((threadsPerBlock-1),ceil((float)maxThreads/(float)(threadsPerBlock-1))), (maxThreads)*sizeof(cuFloatComplex)>> >(Yf, cols-1, rows);
	//	cudaDeviceSynchronize();
	} else {
		dim3 gridDims3(ceil((float)(threadsPerBlock-1)/(float)maxThreads), lenOfBuffer-1);
		shiftOneRow<< <gridDims3, maxThreads, (maxThreads)*sizeof(cuFloatComplex)>> >(Yf, cols-1, rows);
	//	cudaDeviceSynchronize();
	}
	cudaMemcpy(dY, Yf, (cols-1)*(lenOfBuffer-1)*sizeof(*dY), cudaMemcpyDeviceToHost);
	
	if(timerEn){
		finish = clock();
		decode[1] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	cudaFree(Yf);
}



#endif