#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "type_manager.h"
#include <stdio.h>
#include <iostream>

#define CUDA_SYNCHRO(cudaStatus) \
	cudaStatus = cudaDeviceSynchronize(); \
	if (cudaStatus != cudaSuccess) { \
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching element_step_kernel!\n", cudaStatus); \
		goto Error; \
	}

#define CUDA_ERRORCHCK(cudaStatus) \
	cudaStatus = cudaGetLastError(); \
	if (cudaStatus != cudaSuccess) { \
		fprintf(stderr, "element_step_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); \
		goto Error; \
	}

cudaError_t element_step_with_CUDA(FLOAT_TYPE * u, FLOAT_TYPE * v, FLOAT_TYPE * a,
	FLOAT_TYPE * load, FLOAT_TYPE * supports, int * neighbors, FLOAT_TYPE * n_vects, FLOAT_TYPE * K, FLOAT_TYPE * C, FLOAT_TYPE * Mi,
	FLOAT_TYPE * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim, FLOAT_TYPE t_load, FLOAT_TYPE t_max, int maxiter,
	char* outfile, int output_frequency, int gridDim, int blockDim);

__global__ void dof_step_kernel(FLOAT_TYPE * u, FLOAT_TYPE * v, FLOAT_TYPE * a,
	FLOAT_TYPE * load, FLOAT_TYPE * supports, int * neighbors, FLOAT_TYPE * n_vects, FLOAT_TYPE * K, FLOAT_TYPE * C, FLOAT_TYPE * Mi,
	FLOAT_TYPE * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim, FLOAT_TYPE loadfunc);

__global__ void increment(FLOAT_TYPE * u, FLOAT_TYPE * v, FLOAT_TYPE * a, FLOAT_TYPE * u_last, FLOAT_TYPE * v_last, int vdim, FLOAT_TYPE dt);
__global__ void memorize_and_increment(FLOAT_TYPE * u, FLOAT_TYPE * v, FLOAT_TYPE * a, FLOAT_TYPE * u_last, FLOAT_TYPE * v_last, int vdim, FLOAT_TYPE dt);

template<typename T>
T *copy2gpu(T *host_data, int dim){
	cudaError_t cudaStatus;
	T *dev_data;

	// Allocate memory on gpu
	cudaStatus = cudaMalloc((void**)&dev_data, dim * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		cudaFree(dev_data);
	}

	// Copy input vectors from host memory to gpu buffers.
	cudaStatus = cudaMemcpy(dev_data, host_data, dim * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		cudaFree(dev_data);
	}
	return dev_data;
}

template<typename T2>
void copy2cpu(T2 *host_data, T2 *dev_data, int dim){
	// Copy data from gpu to host memory. Host variable is passed already initialized.
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(host_data, dev_data, dim * sizeof(T2), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		free(host_data);
	}
}