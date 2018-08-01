#ifndef cudaEn
	#define cudaEn
#endif

//Shared Memory 
#include "ShMemSymBuff.hpp"
#include <cufft.h>

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
 
//!How to Compile:   nvcc -D prefix=64 gpuLS.cu -lcufft -lrt -o ../build/examples/gpu 
// ./gpu

//LS
//Y = 16 x 1024
//X = 1 x 1023
//H = 16 x 1023
ShMemSymBuff* buffPtr;

using namespace std;

std::string file = "Output.dat";
std::ofstream outfile;
	
//Reads in Vector X from file -> 1xcols
void matrix_readX(complexF* X, int cols){
	ifstream inFile;
	inFile.open(fileNameForX);
	if (!inFile) {
		cerr << "Unable to open file "<< fileNameForX<<", filling in 1+i for x\n";
		float c=1.0f;
		for (int col = 0; col <  cols; col++){
			X[col].real=c;
			X[col].imag=c;
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
	complexF* temp = 0;
	temp=(complexF*)malloc ((cols-1)/2* sizeof (*temp));
	//copy second half to temp
	memmove(temp, &X[(cols+1)/2], (cols-1)/2* sizeof (*X));
	//copy first half to second half
	memmove(&X[(cols-1)/2], X, (cols+1)/2* sizeof (*X));
	//copy temp to first half
	memmove(X, temp, (cols-1)/2* sizeof (*X));
	
	free(temp);
	inFile.close();
}
	
void shiftOneRow(complexF* Y, int cols, int row){
	complexF* Yf = &Y[row*cols];
	//std::cout << "Here...\n";
	complexF* temp = 0;
	temp=(complexF*)malloc ((cols+1)/2* sizeof (*temp));
	//copy second half to temp
	memmove(temp, &Yf[(cols-1)/2], (cols+1)/2* sizeof (*Yf));
	//copy first half to second half
	memmove(&Yf[(cols+1)/2], Yf, (cols-1)/2* sizeof (*Yf));
	//copy temp to first half
	memmove(Yf, temp, (cols+1)/2* sizeof (*Yf));
	
	free(temp);
}	
	
void findDistSqrd(complexF* H, complexF* Hsqrd, int rows, int cols){
	//initialize first row since Hsqrd currently holds X
	for (int j = 0; j<cols; j++){
		//|H|^2 = real^2 + imag^2
		//Sum of |H|^2 is summing all elements in col j
		Hsqrd[j].real = (H[j].real*H[j].real)+ (H[j].imag*H[j].imag);
		Hsqrd[j].imag =0;
	}
	
	for (int i = 1; i<rows; i++){  
		for (int j = 0; j<cols; j++){
			//|H|^2 = real^2 + imag^2
			//Sum of |H|^2 is summing all elements in col j
			Hsqrd[j].real = Hsqrd[j].real+ (H[i*cols + j].real*H[i*cols + j].real)+ (H[i*cols + j].imag*H[i*cols + j].imag);
		}
	}
	
}

//Removes prefix and moves all elements to look like a rows*cols array not rows*(cols+prefix)
__global__ void dropPrefix(complexF* Y, complexF* dY, int rows1, int cols1){
	
	int rows = rows1;
	int cols= cols1;
    
	
    //find my work
    int i = blockIdx.x;
	/*
	if(row!=0){
		return;
	}
	*/
	
	//DROP the prefix
	for(int i =0; i<rows; i++){
		memcpy(&Y[i*cols], &dY[i*(cols+prefix)+ prefix], cols*sizeof(*dY));
	}
	
			
	
}
	
__global__ void findHs(complexF* dY,complexF* dH,complexF* dX,int rows1,int cols1){
	
	//int rows = rows1;
	int cols=cols1;
	
	//find my work
    int row = blockIdx.x;
	
	//Drop first element and copy it into Hconj
	memcpy(&dH[row*(cols-1)], &dY[row*cols+1], (cols-1)* sizeof (*dY));
	
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
	for(int j=0; j<c; j++){
		float fxa = dH[i*c+j].real;
		float fxb = dH[i*c+j].imag;
		float fya = dX[j].real;
		float fyb = dX[j].imag;
		dH[i*c+j].real=((fxa*fya + fxb*fyb)/(fya*fya+fyb*fyb));
		dH[i*c+j].imag= ((fxb*fya - fxa*fyb)/(fya*fya + fyb*fyb));	
	}
	
	//Now dH holds conj H
}
	
//Finds |H|^2 and H*=Hconj, rows=16 cols=1024
void firstVector(complexF* dY, complexF* dH, complexF* dX, int rows, int cols){
	
	//X = 1x1023 -> later can become |H|^2
	complexF* X = 0;
	int sizeX=(cols-1)* sizeof(*X);
	X = (complexF *)malloc(sizeX);
	complexF* H =0;
	H = (complexF *)malloc(sizeX*rows);
	
	//Read in X vector -> 1x1023
	matrix_readX(X, cols-1);
	cudaMemcpy(dX, X, sizeX, cudaMemcpyHostToDevice);
	
	
	// CUFFT plan -> do it one time before?
	//cufftExecC2C(plan, (cufftComplex *)&dY, (cufftComplex *)&dY, CUFFT_FORWARD);
	
	
	//Read in Y with prefix
	buffPtr->readNextSymbolCUDA(dY, 0);
	decode[0]=0;
	//drop the prefix and move into first part of dY
	complexF* Y = 0;
	cudaMalloc((void**)&Y, rows*cols*sizeof(*Y));
	if(prefix>0){
		clock_t start, finish;
		if(timerEn){
			start = clock();
		}
		dropPrefix<< <numOfBlocks, threadsPerBlock >> >(Y, dY, rows, cols);
		if(timerEn){
			finish = clock();
			drop[0] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
		}
	}
	clock_t start, finish;
	if(timerEn){
		start = clock();
	}
	
	//FFT(Y)
	cufftComplex* org = (cufftComplex*)&Y;
	cufftComplex* after = (cufftComplex*)&Y;
	cufftHandle plan;
    cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, org, after, CUFFT_FORWARD);	
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
	
	//H holds Hconj
	cudaMemcpy(H, dH, sizeX*rows, cudaMemcpyDeviceToHost);
	
	//Save |H|^2 into X
	findDistSqrd(H,X,rows, cols-1);
	cudaMemcpy(dX, X, sizeX, cudaMemcpyHostToDevice);
	
	if(timerEn){
		finish = clock();
		decode[0] = decode[0]+ ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	
	//printOutArr(dH, rows, cols-1);
	free(X);
	free(H);
	
	//dH holds H conj
	//dX holds {H^2)	
}

__global__ void doOneSymbol(complexF* Y, complexF* Hconj, complexF* Hsqrd,int rows1, int cols1, int it){
	int rows = rows1;
	int cols= cols1;
    
    //find my work
    int row = blockIdx.x;
	printf("Row: %d\n",row);
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1023
	int i = row;
	//int cp = cols;
	
	
	
	int c = cols-1;
	
	//for(int row=0; row<rows; row++){
		/*
		complexF* Yf = 0;
		Yf = (complexF*)malloc(rows*(cols-1)*sizeof(*Yf));
		memcpy(&Yf[row*(cols-1)], &Y[row*cols+1], (cols-1)* sizeof (*Y));
		//complexF* Yf = &Y[row*cp + 1];
		complexF* temp = 0;
		temp=(complexF*)malloc ((cols+1)/2* sizeof (*temp));
		//copy second half to temp
		memcpy(temp, &Yf[(c-1)/2], (c+1)/2* sizeof (*Yf));
		//copy first half to second half
		memcpy(&Yf[(c+1)/2], Yf, (c-1)/2* sizeof (*Yf));
		//copy temp to first half
		memcpy(Yf, temp, (c+1)/2* sizeof (*Yf));
		
		free(temp);
		*/
	//}
	
	complexF* Yf = 0;
	Yf = (complexF*)malloc(rows*(cols-1)*sizeof(*Yf));
	memcpy(&Yf[row*(cols-1)], &Y[row*cols], (cols-1)* sizeof (*Yf));
	//Calculate product for every element in your row
	for(int j=0; j<cols-1; j++){
		float Yreal = Yf[i*c+j].real;
		float Yimag = Yf[i*c+j].imag;
		float Hreal = Hconj[i*c+j].real;
		float Himag = Hconj[i*c+j].imag;
		//(a+bi)(c+di) = a*c - b*d + (bc + ad)i
		Yf[i*c+j].real=(Yreal*Hreal - Yimag*Himag);
		Yf[i*c+j].imag=(Yreal*Himag + Yimag*Hreal);	
	}
	
	__syncthreads();
	//Find sum of YH* -> 1x1023
	if(row==0){
		for(int r=1; r<rows; r++){
			for(int j=0; j<cols-1; j++){
				Yf[j].real = Yf[j].real + Yf[r*c+j].real;
				Yf[j].imag = Yf[j].imag + Yf[r*c+j].imag;
			}
		}
		
		//Divide YH* / |H|^2
		for(int j=0; j<cols-1; j++){
			Y[j].real = Yf[j].real/Hsqrd[j].real;
			Y[j].imag = Yf[j].imag/Hsqrd[j].real;
			/*
			float fxa = Y[j].real;
			float fxb = Y[j].imag;
			float fya = Hsqrd[j].real;
			float fyb = Hsqrd[j].imag;
			Y[j].real=((fxa*fya + fxb*fyb)/(fya*fya+fyb*fyb));
			Y[j].imag=((fxb*fya - fxa*fyb)/(fya*fya + fyb*fyb));	
			*/
		}
		
	}
	

}

void symbolPreProcess(complexF* Y, complexF* Hconj, complexF* Hsqrd,int rows1, int cols1, int it) {
	int rows = rows1;
	int cols= cols1;
    
    //find my work
    //int row = threadIdx.x;
	
	//Y x conj(H) -> then sum all rows into elements in Hsqrd
	//Y = 16x1024+prefix
	//conjH = 16x1023
	//int i = row;
	//int cp = cols+prefix;
	
	complexF* dY = 0;
	cudaMalloc((void**)&dY, rows*cols*sizeof(*dY));
	if(prefix>0){
		clock_t start, finish;
		if(timerEn){
			start = clock();
		}
		dropPrefix<< <numOfBlocks, threadsPerBlock >> >(dY, Y, rows, cols);
		if(timerEn){
			finish = clock();
			drop[it] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
		}
	}
	clock_t start, finish;
	if(timerEn){
		start = clock();
	}
	//FFT(Y)
	cufftComplex* org = (cufftComplex*)&dY;
	cufftComplex* after = (cufftComplex*)&dY;
	cufftHandle plan;
    cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, org, after, CUFFT_FORWARD);
	
	
	doOneSymbol<< <numOfBlocks, threadsPerBlock >> >(dY, Hconj, Hsqrd, rows, cols, it);
	if(timerEn){
		finish = clock();
		decode[it] = ((float)(finish - start))/(float)CLOCKS_PER_SEC;
	}
	
	
}

__global__ void printDims() {
	int row = blockIdx.x;
	int col = threadIdx.x;
	printf("Block Index: %d\n",row);
	printf("Thread Index: %d\n",col);
}

int main(){
	int rows = numOfRows; // number of vectors
	int cols=dimension;//dimension
	
	printf("CUDA LS: \n");
	printInfo();
	printDims<< <numOfBlocks, threadsPerBlock >> >();
	//dY holds symbol with prefix
	complexF *dY;
	int size = (cols+prefix)*rows* sizeof (*dY);
	cudaMalloc((void**)&dY, size);
	
	//dH (and Hconj) = 16x1023
	complexF* dH = 0;
	size = rows*(cols-1)* sizeof (*dH);
	cudaMalloc((void**)&dH, size);
	
	//X = 1x1023 -> later can become |H|^2
	complexF* dX = 0;
	size = (cols-1)* sizeof (*dX);
	cudaMalloc((void**)&dX, size);
	
	complexF* Yf =0;
	Yf = (complexF*)malloc(size);
	
	//Shared Memory
	string shm_uid = shmemID;
	buffPtr=new ShMemSymBuff(shm_uid, mode);
	
	cufftHandle plan;
    cufftPlan1d(&plan, cols, CUFFT_C2C, rows);
	cufftExecC2C(plan, (cufftComplex *)&dY, (cufftComplex *)&dY, CUFFT_FORWARD);
	
	//Find H* (H conjugate) ->16x1023 and |H|^2 -> 1x1023
	firstVector(dY, dH, dX, rows, cols);
	//dH holds h conj
	//dX holds |H|^2
	
	for(int i=1; i<=numberOfSymbolsToTest; i++){
		if(i==numberOfSymbolsToTest){
			//if last one
			buffPtr->readLastSymbolCUDA(dY);
			/*
			doOneSymbol<< <numOfBlocks, threadsPerBlock >> >(dY, dH, dX, rows, cols, i);
			if(testEn){
				printf("Symbol #%d:\n", i);
				//cuda copy it over
				cudaMemcpy(Yf, dY, size, cudaMemcpyDeviceToHost);
				//printOutArr(Yf, 1, cols-1);
			}
			break;
			*/
		}
		else{
			buffPtr->readNextSymbolCUDA(dY,i);
		}
		//drop prefix - instead I'll just use the format in doOneSymbol
		/*if(prefix>0){
			dropPrefix<< <numOfBlocks, threadsPerBlock >> >(dY, rows, cols);
		}*/
		
		symbolPreProcess(dY, dH, dX, rows, cols, i);
		
		if(testEn){
			//printf("Symbol #%d:\n", i);
			//cuda copy it over
			cudaMemcpy(Yf, dY, size, cudaMemcpyDeviceToHost);
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
