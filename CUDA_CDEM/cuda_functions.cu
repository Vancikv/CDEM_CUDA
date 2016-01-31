#include "cuda_functions.cuh"
#include "aux_functions.h"
#include <fstream>
#include <istream>

cudaError_t element_step_with_CUDA(double * u, double * v, double * a,
	double * load, double * supports, int * neighbors, double * n_vects, double * K, double * C, double * Mi,
	double * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim, 
	double t_load, double t_max, int maxiter,char * outfile, int output_frequency)
{
	// Declare device vars
	double * dev_u, *dev_v, *dev_a, *dev_load, *dev_supports, * dev_n_vects, *dev_K, *dev_C, *dev_Mi,
	* dev_Kc;
	int * dev_neighbors;
	float dt = t_max / maxiter;

	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate GPU buffers.
	cudaStatus = cudaMalloc((void**)&dev_u, 2*n_nds * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_v, 2 * n_nds * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, 2 * n_nds * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_load, 2 * n_nds * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_supports, 2 * n_nds * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_n_vects, 4 * n_nds * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_neighbors, 2 * n_nds * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_K, 64 * n_els * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_C, 8 * n_els * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_Mi, 8 * n_els * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_Kc, 4 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_supports, supports, 2 * n_nds * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_neighbors, neighbors, 2 * n_nds * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_n_vects, n_vects, 4* n_nds * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_K, K, 64 * n_els * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_Mi, Mi, 8 * n_els * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_C, C, 8 * n_els * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_Kc, Kc, 4 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int warps_per_block = 4;
	int threads_per_block = 32*warps_per_block;
	int nblocks = ((n_nds*n_nodedofs) / threads_per_block) + 1;
	dim3 dimBlock(threads_per_block);
	dim3 dimGrid(nblocks);

	int i, j;
	for (i = 0; i < maxiter; i++)
	{
		for (j = 0; j < n_nds*n_nodedofs; j++)
		{
			u[j] += dt*v[j] + 0.5*dt*dt*a[j];
			v[j] += dt*a[j];
		}
		cudaStatus = cudaMemcpy(dev_u, u, 2 * n_nds * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_v, v, 2 * n_nds * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_a, a, 2 * n_nds * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		element_step_kernel <<< dimGrid, dimBlock >>> (dev_u, dev_v, dev_a, dev_load, dev_supports, dev_neighbors,
		dev_n_vects, dev_K, dev_C, dev_Mi, dev_Kc, n_els, n_nds, n_nodedofs, stiffdim, load_function(dt*i/t_load));

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "element_step_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching element_step_kernel!\n", cudaStatus);
			goto Error;
		}

		// Copy output from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(u, dev_u, 2 * n_nds * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(v, dev_v, 2 * n_nds * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(a, dev_a, 2 * n_nds * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		if ((i%output_frequency) == 0)
		{
			char fn[100];
			sprintf(fn,"%s%d.txt",outfile, i);
			std::ofstream f(fn);

			f << n_nds << " " << n_els << " " << i*dt << std::endl;
			for (j = 0; j < n_nds; j++)
			{
				f << "node " << (i + 1) << " x y " << u[2 * j] << " " << u[2*j+1]
					<< " " << v[2 * j] << " " << v[2 * j + 1] << " " << a[2 * j] << " " << a[2 * j + 1] << std::endl;
			}
		}
	}
Error:
	cudaFree(dev_u);
	cudaFree(dev_v);
	cudaFree(dev_a);
	cudaFree(dev_supports);
	cudaFree(dev_neighbors);
	cudaFree(dev_n_vects);
	cudaFree(dev_K);
	cudaFree(dev_Mi);
	cudaFree(dev_C);
	cudaFree(dev_Kc);

	return cudaStatus;
}

__global__ void element_step_kernel(double * u, double * v, double * a, double * load, double * supports, int * neighbors, 
	double * n_vects, double * K, double * C, double * Mi, double * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim, 
	float loadfunc)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x; // thread id - global number of dof
	if (tid < n_nds*n_nodedofs)
	{
		int eid = tid / stiffdim; // global number of element
		int nid = (tid / n_nodedofs) * n_nodedofs; // number of dof 1 of this node
		int ned = tid % stiffdim; // number of dof within element
		int mdim = stiffdim*stiffdim; // number of elements of the stiffness matrix
		int i;
		double kc11 = Kc[0];
		double kc21 = Kc[1];
		double kc12 = Kc[2];
		double kc22 = Kc[3];

		// Element stiffness force:
		double F_k_e = 0;
		for (i = 0; i < stiffdim; i++)
		{
			F_k_e += -K[eid*mdim + i*stiffdim + ned] * u[eid*stiffdim + i];
		}
		// Contact stiffness force:
		double F_k_c = 0;
		for (i = 0; i < 2; i++)
		{
			int nbr = neighbors[nid + i];
			if (nbr != 0)
			{
				double t11 = n_vects[4*(tid/n_nodedofs)+2*i];
				double t12 = n_vects[4 * (tid / n_nodedofs) + 2 * i+1];
				double t21 = -t12;
				double t22 = t11;
				double du_x = u[(nbr - 1)*n_nodedofs] - u[nid];
				double du_y= u[(nbr - 1)*n_nodedofs+1] - u[nid+1];
				if (tid == nid) // X-component
				{
					F_k_c += du_x * (t11*(t11*kc11 + t21*kc21) + t21*(t11*kc12 + t21*kc22)) + du_y * (t12*(t11*kc11 + t21*kc21) + t22*(t11*kc12 + t21*kc22)); // T_T * Kc * T * du_g
				}
				else // Y-component
				{
					F_k_c += du_x * (t11*(t12*kc11 + t22*kc21) + t21*(t12*kc12 + t22*kc22)) + du_y * (t12*(t12*kc11 + t22*kc21) + t22*(t12*kc12 + t22*kc22)); // T_T * Kc * T * du_g
				}
			}
		}
		// Damping force:
		double F_c = -C[tid] * v[tid];
		// Reaction force
		double F_r = supports[tid] * (-F_k_e - F_k_c - F_c - loadfunc*load[tid]);
		a[tid] = Mi[tid] * (F_k_e + F_k_c + F_r + F_c + loadfunc*load[tid]);
	}
}