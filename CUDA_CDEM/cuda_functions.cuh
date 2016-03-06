#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

cudaError_t element_step_with_CUDA(double * u, double * v, double * a,
	double * load, double * supports, int * neighbors, double * n_vects, double * K, double * C, double * Mi,
	double * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim, double t_load, double t_max, int maxiter,
	char* outfile, int output_frequency);

__global__ void element_step_kernel(double * u, double * v, double * a,
	double * load, double * supports, int * neighbors, double * n_vects, double * K, double * C, double * Mi,
	double * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim, double loadfunc);

__global__ void increment(double * u, double * v, double * a, int vdim);

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


double *copy2cpu(double *d_u, int dim);

