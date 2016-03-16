#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

cudaError_t element_step_with_CUDA(float * u, float * v, float * a,
	float * load, float * supports, int * neighbors, float * n_vects, float * K, float * C, float * Mi,
	float * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim, float t_load, float t_max, int maxiter,
	char* outfile, int output_frequency, int gridDim, int blockDim);

__global__ void element_step_kernel(float * u, float * v, float * a,
	float * load, float * supports, int * neighbors, float * n_vects, float * K, float * C, float * Mi,
	float * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim, float loadfunc);

__global__ void increment(float * u, float * v, float * a, float * u_last, float * v_last, int vdim, float dt);
__global__ void memorize_and_increment(float * u, float * v, float * a, float * u_last, float * v_last, int vdim, float dt);

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
T2 *copy2cpu(T2 *dev_data, int dim){
	cudaError_t cudaStatus;
	T2 *host_data;
	host_data = new T2[dim];

	// Copy output vector from gpu buffer to host memory.
	cudaStatus = cudaMemcpy(host_data, dev_data, dim * sizeof(T2), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		free(host_data);
	}
	return host_data;
}