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
 
//!How to Compile:   nvcc ../../examples/gpuLS_cucomplex.cu -lcufft -lrt -o gpu 
// ./gpu

//LS
//Y = 16 x 1024
//X = 1 x 1023
//H = 16 x 1023
ShMemSymBuff* buffPtr;

using namespace std;

std::string file = "Output_gpu.dat";
//std::ofstream outfile;
	
	
//Reads in Vector X from file -> 1xcols
void matrix_readX(cuFloatComplex* X, int cols){
	ifstream inFile;
	inFile.open(fileNameForX);
	if (!inFile) {
		cerr << "Unable to open file "<< fileNameForX<<", filling in 1+i for x\n";
		float c=1.0f;
		for (int col = 0; col <  cols; col++){
			X[col].x=c;
			X[col].y=c;
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

void shiftOneRow(cuFloatComplex* Y, int cols, int row){
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

void dropPrefix(cuFloatComplex *Y, cuFloatComplex *dY, int rows1, int cols1){
	
	int rows = rows1;
	int cols= cols1;	
	for(int i =0; i<rows; i++){
		memcpy(&Y[i*cols], &dY[i*(cols+prefix)+ prefix], cols*sizeof(*dY));
	}		
}

__device__ cuFloatComplex tempDev[numOfBlocks*(threadsPerBlock-1)];
__global__ void reshapeSym(cuFloatComplex* Y, cuFloatComplex* Yf){
	int row = blockIdx.x;
	int col = threadIdx.x;
	tempDev[row*blockDim.x + col] = Yf[col*blockDim.x + row];
	__syncthreads();
	Y[row*blockDim.x + col] = tempDev[row*blockDim.x + col];
	__syncthreads();	
}

__global__ void findHs(cuFloatComplex* dY,cuFloatComplex* dH,cuFloatComplex* dX,int rows1,int cols1){	
	int cols=cols1;
	//find my work
	//Drop first element and copy it into Hconj
	dH[blockIdx.x*blockDim.x + threadIdx.x] = dY[blockIdx.x*(blockDim.x + 1) + threadIdx.x + 1];
	__syncthreads();
	
	//complex division
	//H/X where H = FFT(Y) (w/ dropped first element)
	//Then take conjugate of H
	if (threadIdx.x < (cols-1)) {
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
	//temp[col].x = temp[col].x*temp[col].x + temp[col].y*temp[col].y;
//	temp[threadIdx.x].x = tempLocal.x * tempLocal.x + tempLocal.y * tempLocal.y;
//	__syncthreads();
	
	
	for (int i = 1; i < rows; i = i*2) {
		if (threadIdx.x%(2*i) == 0) {
			temp[threadIdx.x].x += temp[threadIdx.x+i].x;
		}
		__syncthreads();
	}
	
	
	if(threadIdx.x == 0) {
		Hsqrd[blockIdx.x] = temp[threadIdx.x].x;
	/*	
		Hsqrd[blockIdx.x] = 0;
		for (int i = 0; i < numOfBlocks; i++) {
			
			Hsqrd[blockIdx.x] = Hsqrd[blockIdx.x] + H[i*cols + blockIdx.x].x;
		}
		*/
	}
}


void firstVector(cuFloatComplex* dY, cuFloatComplex* dH, cuFloatComplex* dX, float* Hsqrd, int rows, int cols){
	clock_t start, finish;
	//std::cout << "Here...\n";
	//X = 1x1023 -> later can become |H|^2
	cuFloatComplex* X = 0;
	int sizeX=rows*(cols-1)* sizeof(*X);
	X = (cuFloatComplex*)malloc(sizeX);
	//complexF* H =0;
	//H = (complexF *)malloc(sizeX*rows);
	//cudaMalloc((void**)&H, size);
	
	//Read in X vector -> 1x1023
	for (int i = 0; i < rows; i++) {
		//std::cout << "Here...\n";
		matrix_readX(&X[i*(cols-1)], cols-1);
	}
	//std::cout << "Here...\n";
	cudaMemcpy(dX, X, rows*(cols-1)*sizeof(*dX), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();	
	
	// CUFFT plan -> do it one time before?
	
	
	//Read in Y with prefix
	buffPtr->readNextSymbol(dY, 0);
	decode[0]=0;
	//drop the prefix and move into first part of dY
	cuFloatComplex* Y = 0;
	cudaMalloc((void**)&Y, rows*cols*sizeof(*Y));
	cufftHandle plan;
    cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
	if(timerEn){
		start = clock();
	}
	cudaMemcpy(Y, dY, rows*cols*sizeof(*Y), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	if(timerEn){
		finish = clock();
		readT[0] = readT[0] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}

	
	if(timerEn){
		start = clock();
	}
	
	
	//FFT(Y)
//	cufftHandle plan;
//	cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	if(timerEn){
		finish = clock();
		fft[0] = fft[0]+ ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	//find Hconj and Hsqrd
	cuFloatComplex *H = 0;
	cudaMalloc((void**)&H, rows*(cols-1)* sizeof (*H));
	if(timerEn){
		start = clock();
	}
//	dim3 dimBlock(numOfBlocks, threadsPerBlock-1);
	findHs<< <numOfBlocks, threadsPerBlock-1>> >(Y, dH, dX, rows, cols);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
//	reshapeSym<< <numOfBlocks,threadsPerBlock-1 >> >(H, dH);
//	cudaDeviceSynchronize();
	//Save |H|^2 into Hsqrd
	findDistSqrd<< <threadsPerBlock-1, numOfBlocks, numOfBlocks*sizeof(cuFloatComplex)>> >(dX,Hsqrd,rows, cols-1);
	cudaDeviceSynchronize();
	
	if(timerEn){
		finish = clock();
		decode[0] = decode[0] + ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	std::string file = "Chan_est.dat";
	cuFloatComplex* Yf;
	Yf = (cuFloatComplex*)malloc(rows*(cols-1)*sizeof(*Yf));
	cudaMemcpy(Yf, dH, rows*(cols-1)*sizeof(*dH), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	std::cout << "After Chan Est...\n";
	outfile.open(file.c_str(), std::ofstream::binary);
	outfile.write((const char*)Yf, rows*(cols-1)*sizeof(*Yf));
	outfile.close();
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	
	cudaMemcpy(Yf, Hsqrd, (cols-1)*sizeof(*Hsqrd), cudaMemcpyDeviceToHost);
	std::cout << "After Squared...\n";
	file = "Dist_sqrd.dat";
	outfile.open(file.c_str(), std::ofstream::binary);
	outfile.write((const char*)Yf, rows*(cols-1)*sizeof(*Hsqrd));
	outfile.close();
	
	free(X);
	cudaFree(H);
	
	//dH holds H conj
	//dX holds {H^2)	
}


__global__ void doOneSymbol(cuFloatComplex* Y, cuFloatComplex* Hconj, cuFloatComplex* Yf, int rows1, int cols1){
	int rows = rows1;
	int cols= cols1;
    
    //find my work 
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1023
	int row = blockIdx.x;
	int j = threadIdx.x;
	Yf[row*blockDim.x + j] = Y[row*(blockDim.x+1) + j + 1];
	__syncthreads();
	
	if (j < cols-1) {
		Yf[row*blockDim.x + j] = cuCmulf(Yf[row*blockDim.x + j],Hconj[row*blockDim.x + j]);
	}
	__syncthreads();
}


__global__ void combineForMRC(cuFloatComplex *Y, float *Hsqrd, int rows, int cols) {
	
	int row = blockIdx.x;
	int col = threadIdx.x;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	extern __shared__ cuFloatComplex temp[];
	int sid = threadIdx.x*cols + blockIdx.x;
	temp[col] = Y[sid];
	
	for (int i = 1; i < rows; i = i*2) {
		if (threadIdx.x%(2*i) == 0) {
			temp[col] = cuCaddf(temp[col],temp[col+i]);
		}
		__syncthreads();
	}
	
	if (col == 0) {
		Y[row].x = temp[col].x/Hsqrd[row];
		Y[row].y = temp[col].y/Hsqrd[row];
		__syncthreads();
	}
	
	/*
	if (threadIdx.x == 0) {
		for (int i = 1; i < numOfRows; i++) {
			Y[blockIdx.x] = cuCaddf(Y[blockIdx.x],Y[i*cols + blockIdx.x]);
		}
		Y[blockIdx.x].x = Y[blockIdx.x].x/Hsqrd[blockIdx.x];
		Y[blockIdx.x].y = Y[blockIdx.x].y/Hsqrd[blockIdx.x];
	}
	*/
	
}


void symbolPreProcess(cuFloatComplex *Y, cuFloatComplex *Hconj, float *Hsqrd,int rows1, int cols1, int it) {
	int rows = rows1;
	int cols= cols1;

	clock_t start, finish;
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1023
	if(timerEn){
		start = clock();
	}
	
	cuFloatComplex* dY = 0;
	cudaMalloc((void**)&dY, rows*cols*sizeof(*dY));
	cudaMemcpy(dY, Y, rows*cols*sizeof(*dY), cudaMemcpyHostToDevice);
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
	cufftExecC2C(plan, (cufftComplex *)dY, (cufftComplex *)dY, CUFFT_FORWARD);
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
	doOneSymbol<< <numOfBlocks, threadsPerBlock-1>> >(dY, Hconj, Yf, rows, cols);
	cudaDeviceSynchronize();
	combineForMRC<< <threadsPerBlock-1, numOfBlocks, numOfBlocks*sizeof(cuFloatComplex)>> >(Yf, Hsqrd, rows, cols-1);
	cudaMemcpy(Y, Yf, rows*(cols-1)*sizeof(*Y), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	/*
	if(timerEn){
		start = clock();
	}
	*/
	/*
	for(int r=1; r<rows; r++){
		for(int j=0; j<cols-1; j++){
			Y[j]= cuCaddf(Y[j],Y[r*(cols-1)+j]);
		}
	}
	
	//Divide YH* / |H|^2
	for(int j=0; j<cols-1; j++){
		Y[j].x = Y[j].x/Hsqrd[j];
		Y[j].y = Y[j].y/Hsqrd[j];
	}
	*/
	
	shiftOneRow(Y, cols-1, 0);
	
	if(timerEn){
		finish = clock();
		decode[it] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	cudaFree(dY);
	cudaFree(Yf);
	cudaDeviceSynchronize();
}



int main(){
	int rows = numOfRows; // number of vectors
	int cols=dimension;//dimension
	cudaSetDevice(0);
	//printf("CUDA LS: \n");
	//printInfo();
	//dY holds symbol with prefix
	cuFloatComplex *dY = 0;
	dY = (cuFloatComplex*)malloc(rows*cols* sizeof (*dY));
	
	float *Hsqrd = 0;
	cudaMalloc((void**)&Hsqrd, (cols-1)* sizeof (*Hsqrd));
	
	//dH (and Hconj) = 16x1023
	cuFloatComplex *dH = 0;
	cudaMalloc((void**)&dH, rows*(cols-1)* sizeof (*dH));
	
	//X = 1x1023 -> later can become |H|^2
	cuFloatComplex *dX = 0;
	cudaMalloc((void**)&dX, rows*(cols-1)* sizeof (*dX));
	
	cuFloatComplex *Yf = 0;
	Yf = (cuFloatComplex*)malloc((cols-1)* sizeof (*Yf));
	
	//Shared Memory
	string shm_uid = shmemID;
	buffPtr=new ShMemSymBuff(shm_uid, mode);
	
	
	/*cufftComplex *temp = 0;
	cudaMalloc((void**)&temp, rows*cols* sizeof (*temp));
	cufftHandle plan;
    cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)&temp, (cufftComplex *)&temp, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	cudaFree(temp);
	*/
	//Find H* (H conjugate) ->16x1023 and |H|^2 -> 1x1023
	firstVector(dY, dH, dX, Hsqrd, rows, cols);
	//dH holds h conj
	//dX holds |H|^2
	
	for(int i=1; i<numberOfSymbolsToTest; i++){
		if(i==numberOfSymbolsToTest-1){
			//if last one
			buffPtr->readLastSymbol(dY);
		}
		else{
			buffPtr->readNextSymbol(dY,i);
			/*
			if (i == 1) {
				std::string file = "Sym_copy.dat";
//				cuFloatComplex Yf_[rows*(cols+prefix)];
				cuFloatComplex *Yf_;
				Yf_ = (cuFloatComplex*)malloc(rows*(cols)*sizeof(*Yf_));
				memcpy(Yf_, dY, rows*(cols)*sizeof(*Yf_));
				//cudaDeviceSynchronize();
				//std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
				std::cout << "Copied back to CPU...\n";
				//printOutArr(Yf_,1,cols+prefix);
					for (int j = 0; j < rows*(cols); j = j + cols) {
						cout << "(" << Yf_[j].x << ", " << Yf_[j].y << ")\n";
					}
				outfile.open(file.c_str(), std::ofstream::binary);
				outfile.write((const char*)Yf_, rows*(cols)*sizeof(*Yf_));
				outfile.close();
			}
			*/
			
		}
		symbolPreProcess(dY, dH, Hsqrd, rows, cols, i);
		
		if(testEn){
			//printf("Symbol #%d:\n", i);
			//cuda copy it over
			memcpy(Yf, dY, (cols-1)* sizeof (*Yf));
			if (i <= 1) {
				outfile.open(file.c_str(), std::ofstream::binary | std::ofstream::trunc);
			} else {
				outfile.open(file.c_str(), std::ofstream::binary | std::ofstream::app);
			}
			outfile.write((const char*)Yf, (cols-1)*sizeof(*Yf));
			outfile.close();
			//printOutArr(Yf, 1, cols-1);
		}
		
		
	}
	
	free(Yf);
	cudaFree(dY);
	cudaFree(dH);
	cudaFree(dX);
	//delete buffPtr;
	
	if(timerEn) {
		printTimes(true);
		storeTimes(false);
	}
	return 0;

}