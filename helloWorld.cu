#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

using namespace std;
#define N 1023

struct complexF{
	float real;
	float imag;
};

__global__ void addNums(cuFloatComplex *a, cuFloatComplex *b, cuFloatComplex *c) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < N)
		c[id] = cuCaddf(a[id],b[id]);
}

int main() {
	int count;
	cudaGetDeviceCount(&count);
	cout << "\n\nThe number of devices supported are " << count << endl;
	cudaDeviceProp devProp;
	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&devProp, i);
		cout << "Device ID: " << devProp.name << endl;
		cout << "Total global memory: " << devProp.totalGlobalMem << endl;
		cout << "Memory pitch: " << devProp.memPitch << endl;
		cout << "Total constant memory: " << devProp.totalConstMem << endl;
		cout << "Number of Processor(s): " << devProp.multiProcessorCount << endl;
		//cout << "Number of register(s) per processor: " << devProp.regsPerMultiprocessor << endl;
		cout << "Number of thread(s) per processor: " << devProp.maxThreadsPerMultiProcessor << endl;
		cout << "Number of thread(s) per block: " << devProp.maxThreadsPerBlock << endl;
	}
	

	cuFloatComplex a[N], b[N], ans[N];
	cuFloatComplex *dev_a, *dev_b, *c;
	cudaMalloc((void**)&dev_a, N*sizeof(*dev_a));
	cudaMalloc((void**)&dev_b, N*sizeof(*dev_b));
	cudaMalloc((void**)&c, N*sizeof(*c));
	for (int i = 0; i < N; i++) {
		a[i].x = i;
		a[i].y = i * 3;
		b[i].x = i;
		b[i].y = i * 3;
	}
	cudaMemcpy(dev_a, &a, N * sizeof(*dev_a), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, N * sizeof(*dev_b), cudaMemcpyHostToDevice);
	addNums <<<N,N>>>(dev_a, dev_b, c);
	cudaMemcpy(ans, c, N * sizeof(*c), cudaMemcpyDeviceToHost);
	cout << "\nThe answer is ";
	for (int i = 0; i < N; i = i + 100) {
		cout << "\n" << a[i].x << "+" << b[i].x << "=" << ans[i].x << ", ";
		cout << "\n" << a[i].y << "+" << b[i].y << "=" << ans[i].y << ", ";
	}
	return 0;
}