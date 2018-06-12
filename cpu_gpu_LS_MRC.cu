#ifndef cudaEn
	#define cudaEn
#endif

//Shared Memory 
#include "ShMemSymBuff_cucomplex.hpp"
#include <cufft.h>
#include <cuComplex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>

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

std::string file = "Output.dat";
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

__global__ void findHs(cuFloatComplex* dY,cuFloatComplex* dH,cuFloatComplex* dX,int rows1,int cols1){
	
	//int rows = rows1;
	int cols=cols1;
	int rows = rows1;
	//find my work
	//Drop first element and copy it into Hconj
	for (int i = 0; i < rows; i++) {
		memcpy(&dH[i*(cols-1)], &dY[i*cols+1], (cols-1)* sizeof (*dY));
	}
	
	//complex division
	//H/X where H = FFT(Y) (w/ dropped first element)
	//Then take conjugate of H
	int i = blockIdx.x;
	int j = threadIdx.x;
	//for(int j=0; j<c; j++){
	if (j < cols) {
	//	dH[i*blockDim.x + j] = dY[i*blockDim.x + j + 1];
		dH[i*blockDim.x + j] = cuConjf(cuCdivf(dH[i*blockDim.x + j], dX[j]));
	}
	//}
	
	//Now dH holds conj H
}

void findDistSqrd(cuFloatComplex* H, float* Hsqrd, int rows, int cols){
	//initialize first row since Hsqrd currently holds X
	for (int j = 0; j<cols; j++){
		//|H|^2 = real^2 + imag^2
		//Sum of |H|^2 is summing all elements in col j
		Hsqrd[j] = cuCabsf(H[j])*cuCabsf(H[j]);
		//Hsqrd[j].y = 0;
	}
	
	for (int i = 1; i<rows; i++){  
		for (int j = 0; j<cols; j++){
			//|H|^2 = real^2 + imag^2
			//Sum of |H|^2 is summing all elements in col j
			Hsqrd[j] = Hsqrd[j] + cuCabsf(H[i*cols + j])*cuCabsf(H[i*cols + j]);
		}
	}
	
}

void firstVector(cuFloatComplex* dY, cuFloatComplex* dH, cuFloatComplex* dX, float* Hsqrd, int rows, int cols){
	
	//X = 1x1023 -> later can become |H|^2
	cuFloatComplex* X = 0;
	int sizeX=(cols-1)* sizeof(*X);
	X = (cuFloatComplex*)malloc(sizeX);
	//complexF* H =0;
	//H = (complexF *)malloc(sizeX*rows);
	//cudaMalloc((void**)&H, size);
	
	//Read in X vector -> 1x1023
	matrix_readX(X, cols-1);
	cudaMemcpy(dX, X, sizeX, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	
	
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
	cudaMemcpy(Y, dY, rows*cols*sizeof(*Y), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	
	/*
	if(prefix>0){
		clock_t start, finish;
		if(timerEn){
			start = clock();
		}
		dropPrefix(Y, dY, rows, cols);
		cudaDeviceSynchronize();
		if(timerEn){
			finish = clock();
			drop[0] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
		}
	}
	*/
	
		//cufftExecC2C(plan, (cufftComplex *)dY, (cufftComplex *)dY, CUFFT_FORWARD);
	
	clock_t start, finish;
	if(timerEn){
		start = clock();
	}
	
	
	//FFT(Y)
//	cufftHandle plan;
//	cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)Y, (cufftComplex *)Y, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	/*
	int c = cols-1;
	for(int row=0; row<rows; row++){
		complexF* Yf = &dY[row*c];
		complexF* temp = 0;
		temp=(complexF*)malloc ((cols+1)/2* sizeof (*temp));
		//copy second half to temp
		memcpy(temp, &Yf[(c-1)/2], (c+1)/2* sizeof (*Yf));
		//copy first half to second half
		memcpy(&Yf[(c+1)/2], Yf, (c-1)/2* sizeof (*Yf));
		//copy temp to first half
		memcpy(Yf, temp, (c+1)/2* sizeof (*Yf));
		
		free(temp);
	}
	*/
	
	//find Hcon and Hsqrd
	findHs<< <numOfBlocks, threadsPerBlock-1 >> >(Y, dH, dX, rows, cols);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	cuFloatComplex *H = 0;
	H = (cuFloatComplex*)malloc(rows*(cols-1)*sizeof(*H));
	cudaMemcpy(H, dH, rows*(cols-1)*sizeof(*dH), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

	//H holds Hconj
	//cudaMemcpy(H, dH, sizeX*rows, cudaMemcpyDeviceToHost);
	
	//Save |H|^2 into X
	findDistSqrd(H,Hsqrd,rows, cols-1);
	
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
	/*
	memcpy(Yf, Hsqrd, (cols-1)*sizeof(*Hsqrd));
	std::cout << "After Squared...\n";
	file = "Dist_sqrd.dat";
	outfile.open(file.c_str(), std::ofstream::binary);
	outfile.write((const char*)Yf, (cols-1)*sizeof(*Yf));
	outfile.close();
	*/
	
	if(timerEn){
		finish = clock();
		decode[0] = decode[0]+ ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	free(X);
	free(Yf);
	free(H);
	
	//dH holds H conj
	//dX holds {H^2)	
}


__global__ void doOneSymbol(cuFloatComplex* Y, cuFloatComplex* Hconj, int rows1, int cols1){
	int rows = rows1;
	int cols= cols1;
    
    //find my work 
    int row = blockIdx.x;
	//printf("Row: %d\n",row);
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1023
	int i = row;
	int j = threadIdx.x;
	//int cp = cols;
	
	
	
	int c = cols-1;
	
	cuFloatComplex* Yf = 0;
	Yf = (cuFloatComplex*)malloc(rows*(cols-1)*sizeof(*Yf));
	/*
	extern __shared__ cuFloatComplex temp[];
	int tid = threadIdx.x;
	if (j > 0) {
		temp[tid] = Y[i*c + j];
	}
	__syncthreads();
	
	if (j < rows*(cols-1)) {
		Yf[i*c + j] = temp[tid];
		temp[tid].x = 0;
		temp[tid].y = 0;
	}
	*/
	/*
	for (int i = 0; i < rows; i++) {
		memcpy(&Yf[i*(cols-1)], &Y[i*cols+1], (cols-1)* sizeof (*Yf));
	}
	*/
	if (j < cols-1) {
		Yf[i*c + j] = Y[i*c + j + 1];
		Yf[i*c+j] = cuCmulf(Yf[i*c+j],Hconj[i*c+j]);
		Y[i*c + j] = Yf[i*c + j];
	}
	__syncthreads();
	free(Yf);
	//free(temp);
}

void symbolPreProcess(cuFloatComplex *Y, cuFloatComplex *Hconj, float *Hsqrd,int rows1, int cols1, int it) {
	int rows = rows1;
	int cols= cols1;	
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1023
	
	cuFloatComplex* dY = 0;
	cudaMalloc((void**)&dY, rows*cols*sizeof(*dY));
	cudaMemcpy(dY, Y, rows*cols*sizeof(*Y), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	std::cout << "Symbol " << it << ": " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	
	if (it == 1) {
		std::string file = "Prefix_drop.dat";
		cuFloatComplex *Yf;
		Yf = (cuFloatComplex*)malloc(rows*cols*sizeof(*Yf));
		cudaMemcpy(Yf, dY, rows*cols*sizeof(*Yf), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
		std::cout << "\n After Prefix drop:\n";
		for (int j = 0; j < rows*(cols); j = j + cols) {
			cout << "(" << Yf[j].x << ", " << Yf[j].y << ")\n";
		}
		
		outfile.open(file.c_str(), std::ofstream::binary);
		outfile.write((const char*)Yf, rows*(cols)*sizeof(*Yf));
		outfile.close();
		free(Yf);
	}
	
	clock_t start, finish;
	if(timerEn){
		start = clock();
	}
	
	//FFT(Y)
	cufftHandle plan;
    cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)dY, (cufftComplex *)dY, CUFFT_FORWARD);
	
	
	if (it == 1) {
		std::string file = "FFT_Out.dat";
		cuFloatComplex* Yf;
		Yf = (cuFloatComplex*)malloc(rows*cols*sizeof(*Yf));
		cudaMemcpy(Yf, dY, rows*cols*sizeof(*Yf), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
		std::cout << "After FFT...\n";
		outfile.open(file.c_str(), std::ofstream::binary);
		outfile.write((const char*)Yf, rows*(cols)*sizeof(*Yf));
		outfile.close();
		free(Yf);
	}
	
	doOneSymbol<< <numOfBlocks, threadsPerBlock>> >(dY, Hconj, rows, cols);
	cudaDeviceSynchronize();
	cudaMemcpy(Y, dY, rows*cols*sizeof(*Y), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(int r=1; r<rows; r++){
		for(int j=0; j<cols-1; j++){
			Y[j]= cuCaddf(Y[j],Y[r*cols+j]);
		}
	}
	
	//Divide YH* / |H|^2
	for(int j=0; j<cols-1; j++){
		Y[j].x = Y[j].x/Hsqrd[j];
		Y[j].y = Y[j].y/Hsqrd[j];
	}
	
	
	shiftOneRow(Y, cols-1, 0);
	
	if(timerEn){
		finish = clock();
		decode[it] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}	
	cudaFree(dY);
	cudaDeviceSynchronize();
}



int main(){
	int rows = numOfRows; // number of vectors
	int cols=dimension;//dimension
	cudaSetDevice(0);
	printf("CUDA LS: \n");
	printInfo();
	//dY holds symbol with prefix
	cuFloatComplex *dY = 0;
	dY = (cuFloatComplex*)malloc(rows*cols* sizeof (*dY));
	
	float *Hsqrd = 0;
	Hsqrd = (float*)malloc((cols-1)*sizeof (*Hsqrd));
	
	//dH (and Hconj) = 16x1023
	cuFloatComplex *dH = 0;
	cudaMalloc((void**)&dH, rows*(cols-1)* sizeof (*dH));
	
	//X = 1x1023 -> later can become |H|^2
	cuFloatComplex *dX = 0;
	cudaMalloc((void**)&dX, (cols-1)* sizeof (*dX));
	
	cuFloatComplex *Yf = 0;
	Yf = (cuFloatComplex*)malloc((cols-1)* sizeof (*Yf));
	
	//Shared Memory
	string shm_uid = shmemID;
	buffPtr=new ShMemSymBuff(shm_uid, mode);
	
	/*
	cufftComplex *temp = 0;
	cudaMalloc((void**)&temp, rows*cols* sizeof (*temp));
	cufftHandle plan;
    cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)&temp, (cufftComplex *)&temp, CUFFT_FORWARD);
	cudaDeviceSynchronize();
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
	if(timerEn)
		printTimes(true);
	return 0;

}