#ifndef cudaEn
	#define cudaEn
#endif

//Shared Memory 
#include "ShMemSymBuff_cucomplex.hpp"
#include <cufft.h>
#include <cuComplex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
	
	//find my work
    int row = blockIdx.x;
	int col = threadIdx.x;
	
	//Drop first element and copy it into Hconj
//	memcpy(&dH[row*(cols-1)], &dY[row*cols+1], (cols-1)* sizeof (*dY));
	
	dH[row*(cols-1) + col] = dY[row*cols + col + 1];
	int c = cols-1;
	
	//shiftOneRow(dH, cols-1, row);
	/*
	complexF* Yf = &dH[row*c];
	complexF* temp = 0;
	temp=(complexF*)malloc ((c+1)/2* sizeof (*temp));
	//copy second half to temp
	memcpy(temp, &Yf[(c-1)/2], (c+1)/2* sizeof (*Yf));
	//copy first half to second half
	memcpy(&Yf[(c+1)/2], Yf, (c-1)/2* sizeof (*Yf));
	//copy temp to first half
	memcpy(Yf, temp, (c+1)/2* sizeof (*Yf));
	
	free(temp);
	*/
	//shift the row
	/*
	complexF* Yf = &dY[row];
	complexF temp[(dimension-1)/2];
	//copy first half to temp
	memcpy(temp, Yf, c/2* sizeof (*Yf));
	//copy second half to first half
	memcpy(Yf, &Yf[cols/2], c/2* sizeof (*Yf));
	//copy first half to second
	memcpy(&Yf[cols/2], temp, c/2* sizeof (*Yf));
	*/
	
	//complex division
	//H/X where H = FFT(Y) (w/ dropped first element)
	//Then take conjugate of H
	int i=row;
	int j = col;
	//for(int j=0; j<c; j++){
		if (j < c){
			dH[i*c+j] = cuConjf(cuCdivf(dH[i*c+j], dX[j]));
			/*
			float fxa = dH[i*c+j].real;
			float fxb = dH[i*c+j].imag;
			float fya = dX[j].real;
			float fyb = dX[j].imag;
			dH[i*c+j].real=((fxa*fya + fxb*fyb)/(fya*fya+fyb*fyb));
			dH[i*c+j].imag= ((fxb*fya - fxa*fyb)/(fya*fya + fyb*fyb));
			*/
		}
	//}
	
	//Now dH holds conj H
}

__global__ void findDistSqrd(cuFloatComplex* H, cuFloatComplex* Hsqrd, int rows, int cols){
	//initialize first row since Hsqrd currently holds X
	__shared__ float temp[threadsPerBlock-1];
	int j = threadIdx.x;
	int i = blockIdx.x;
	//for (int j = 0; j<cols; j++){
		//|H|^2 = real^2 + imag^2
		//Sum of |H|^2 is summing all elements in col j
	if (i == 0) {
		Hsqrd[j].x = cuCabsf(H[j])*cuCabsf(H[j]);//(H[j].real*H[j].real)+ (H[j].imag*H[j].imag);
		Hsqrd[j].y = 0;
	}
	//}
	
	for (int i = 1; i<rows; i++){ 
		temp[j] = cuCabsf(H[i*blockDim.x + j])*cuCabsf(H[i*blockDim.x + j]);
		__syncthreads();
		//for (int j = 0; j<cols; j++){
			//|H|^2 = real^2 + imag^2
			//Sum of |H|^2 is summing all elements in col j
		Hsqrd[j].x = Hsqrd[j].x + temp[j];
		__syncthreads();
		//}
	}
	
}

void firstVector(cuFloatComplex* dY, cuFloatComplex* dH, cuFloatComplex* dX, int rows, int cols){
	
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
	findHs<< <numOfBlocks, threadsPerBlock >> >(Y, dH, dX, rows, cols);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	
	//H holds Hconj
	//cudaMemcpy(H, dH, sizeX*rows, cudaMemcpyDeviceToHost);
	
	//Save |H|^2 into X
	findDistSqrd<< <numOfBlocks, threadsPerBlock-1 >> >(dH,dX,rows, cols-1);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	
	if(timerEn){
		finish = clock();
		decode[0] = decode[0]+ ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	
	//printOutArr(dH, rows, cols-1);
	
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
	free(X);
	free(Yf);
	//free(H);
	
	//dH holds H conj
	//dX holds {H^2)	
}


__global__ void doOneSymbol(cuFloatComplex* Y, cuFloatComplex* Hconj, cuFloatComplex* Hsqrd,int rows1, int cols1, int it){
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
	__shared__ cuFloatComplex temp[numOfBlocks*(threadsPerBlock-1)];
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
	__syncthreads();
	//memcpy(&Yf[row*(cols-1)], &Y[row*cols], (cols-1)* sizeof (*Yf));
	//Calculate product for every element in your row
	//for(int j=0; j<cols-1; j++){
	if (j < cols-1) {
		Yf[i*c+j] = cuCmulf(Yf[i*c+j],Hconj[i*c+j]);
		/*
		float Yreal = Yf[i*c+j].real;
		float Yimag = Yf[i*c+j].imag;
		float Hreal = Hconj[i*c+j].real;
		float Himag = Hconj[i*c+j].imag;
		//(a+bi)(c+di) = a*c - b*d + (bc + ad)i
		Yf[i*c+j].real=(Yreal*Hreal - Yimag*Himag);
		Yf[i*c+j].imag=(Yreal*Himag + Yimag*Hreal);	
		*/
	}
	//}
	
	__syncthreads();
	//Find sum of YH* -> 1x1023
	if(row==0 and (j < cols-1)){
		for(int r=1; r<rows; r++){
			//for(int j=0; j<cols-1; j++){
				temp[j] = cuCaddf(temp[j],Yf[r*c+j]);
				/*
				Yf[j].real = Yf[j].real + Yf[r*c+j].real;
				Yf[j].imag = Yf[j].imag + Yf[r*c+j].imag;
				*/
			//}
		}
		__syncthreads();
		
		//Divide YH* / |H|^2
		//for(int j=0; j<cols-1; j++){
			Y[j].x = temp[j].x/Hsqrd[j].x;
			Y[j].y = temp[j].y/Hsqrd[j].x;
			/*
			float fxa = Y[j].real;
			float fxb = Y[j].imag;
			float fya = Hsqrd[j].real;
			float fyb = Hsqrd[j].imag;
			Y[j].real=((fxa*fya + fxb*fyb)/(fya*fya+fyb*fyb));
			Y[j].imag=((fxb*fya - fxa*fyb)/(fya*fya + fyb*fyb));	
			*/
		//}
		
	}
	__syncthreads();
	free(Yf);

}



void symbolPreProcess(cuFloatComplex *Y, cuFloatComplex *Hconj, cuFloatComplex *Hsqrd,int rows1, int cols1, int it) {
	int rows = rows1;
	int cols= cols1;
    
    //find my work
    //int row = threadIdx.x;
	
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1023
	//int i = row;
	//int cp = cols+prefix;
	
	cuFloatComplex* dY = 0;
	cudaMalloc((void**)&dY, rows*cols*sizeof(*dY));
	cudaMemcpy(dY, Y, rows*cols*sizeof(*Y), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	std::cout << "Symbol " << it << ": " << cudaGetErrorString(cudaGetLastError()) << std::endl;
	/*
	if(prefix>0){
		clock_t start, finish;
		if(timerEn){
			start = clock();
		}
		dropPrefix(dY, Y, rows, cols);
		cudaDeviceSynchronize();
		if(timerEn){
			finish = clock();
			drop[it] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
		}
	}
	*/

	
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
	//cufftComplex* org = (cufftComplex*)&dY;
	//cufftComplex* after = (cufftComplex*)&dY;
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
	
	doOneSymbol<< <numOfBlocks, threadsPerBlock >> >(dY, Hconj, Hsqrd, rows, cols, it);
	cudaDeviceSynchronize();
	cudaMemcpy(Y, dY, (cols-1)*sizeof(*Y), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
//	std::cout << "Symbol " << it << ": " << cudaGetErrorString(cudaGetLastError()) << std::endl;
	if(timerEn){
		finish = clock();
		decode[it] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}	
	cudaFree(dY);
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
	firstVector(dY, dH, dX, rows, cols);
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
					for (int j = 0; j < rows*(cols); j = j + cols+prefix) {
						cout << "(" << Yf_[j].x << ", " << Yf_[j].y << ")\n";
					}
				outfile.open(file.c_str(), std::ofstream::binary);
				outfile.write((const char*)Yf_, rows*(cols)*sizeof(*Yf_));
				outfile.close();
			}
		}
		symbolPreProcess(dY, dH, dX, rows, cols, i);
		
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