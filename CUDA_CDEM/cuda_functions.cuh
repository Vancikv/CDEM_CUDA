#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t element_step_with_CUDA(double * u, double * v, double * a,
	double * load, double * supports, int * neighbors, double * n_vects, double * K, double * C, double * Mi,
	double * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim, double t_load, double t_max, int maxiter);

__global__ void element_step_kernel(double * u, double * v, double * a,
	double * load, double * supports, int * neighbors, double * n_vects, double * K, double * C, double * Mi,
	double * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim, float loadfunc);