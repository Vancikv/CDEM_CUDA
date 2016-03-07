#include "cuda_functions.cuh"
#include "aux_functions.h"
#include <fstream>
#include <istream>

cudaError_t element_step_with_CUDA(double * u, double * v, double * a,
	double * load, double * supports, int * neighbors, double * n_vects, double * K, double * C, double * Mi,
	double * Kc, int n_els, int n_nds, int n_nodedofs, int stiffdim,
	double t_load, double t_max, int maxiter, char * outfile, int output_frequency)
{
	// Declare device vars
	double * dev_u, *dev_v, *dev_a, *dev_load, *dev_supports, *dev_n_vects, *dev_K, *dev_C, *dev_Mi,
		*dev_Kc;
	double * dev_u_last, *dev_v_last;
	int * dev_neighbors;
	double dt = t_max / maxiter;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	dev_u = copy2gpu(u, 2 * n_nds);
	dev_v = copy2gpu(v, 2 * n_nds);
	dev_a = copy2gpu(a, 2 * n_nds);
	dev_u_last = copy2gpu(u, 2 * n_nds);
	dev_v_last = copy2gpu(v, 2 * n_nds);
	dev_load = copy2gpu(load, 2 * n_nds);
	dev_supports = copy2gpu(supports, 2 * n_nds);
	dev_n_vects = copy2gpu(n_vects, 4 * n_nds);
	dev_neighbors = copy2gpu(neighbors, 2 * n_nds);
	dev_K = copy2gpu(K, stiffdim * stiffdim * n_els);
	dev_C = copy2gpu(C, stiffdim * n_els);
	dev_Mi = copy2gpu(Mi, stiffdim * n_els);
	dev_Kc = copy2gpu(Kc, 4);

	int warps_per_block = 4;
	int threads_per_block = 32*warps_per_block;
	int nblocks = ((n_nds*n_nodedofs) / threads_per_block) + 1;
	dim3 dimBlock(threads_per_block,1,1);
	dim3 dimGrid(nblocks,1,1);

	int i, j;
	for (i = 1; i <= maxiter; i++)
	{
		memorize_and_increment << <dimGrid, dimBlock >> >(dev_u, dev_v, dev_a, dev_u_last, dev_v_last, n_nodedofs*n_nds, dt);

		if (i > 1) // Relaxation step
		{
			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching element_step_kernel!\n", cudaStatus);
				goto Error;
			}

			element_step_kernel << < dimGrid, dimBlock >> > (dev_u, dev_v, dev_a, dev_load, dev_supports, dev_neighbors,
				dev_n_vects, dev_K, dev_C, dev_Mi, dev_Kc, n_els, n_nds, n_nodedofs, stiffdim, load_function(dt*i / t_load));

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

			increment << <dimGrid, dimBlock >> >(dev_u, dev_v, dev_a, dev_u_last, dev_v_last, n_nodedofs*n_nds, dt);
		}
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching element_step_kernel!\n", cudaStatus);
			goto Error;
		}

		element_step_kernel << < dimGrid, dimBlock >> > (dev_u, dev_v, dev_a, dev_load, dev_supports, dev_neighbors,
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


		if (((i%output_frequency) == 0) || (i == 1))
		{
			// Copy output from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(u, dev_u, 2 * n_nds * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}
			// Copy output from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(v, dev_v, 2 * n_nds * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}
			// Copy output from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(a, dev_a, 2 * n_nds * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}
			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching element_step_kernel!\n", cudaStatus);
				goto Error;
			}
			char fn[100];
			sprintf(fn,"%s%d.txt",outfile, i);
			std::ofstream f(fn);

			f << n_nds << " " << n_els << " " << i*dt << " " << load_function(dt*i / t_load) << std::endl;
			for (j = 0; j < n_nds; j++)
			{
				f << "node " << (j + 1) << " x y " << u[2 * j] << " " << u[2*j+1]
					<< " " << v[2 * j] << " " << v[2 * j + 1] << " " << a[2 * j] << " " << a[2 * j + 1] << std::endl;
			}
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU solve time %3.5f[s]\n", elapsedTime / 1000);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return cudaStatus;

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
	double loadfunc)
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

__global__ void memorize_and_increment(double * u, double * v, double * a, double * u_last, double * v_last, int vdim, double dt)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x; // thread id - global number of dof
	if (tid < vdim)
	{
		u_last[tid] = u[tid];
		u[tid] += dt*v[tid] + 0.5*dt*dt*a[tid];
		v_last[tid] = v[tid];
		v[tid] += dt*a[tid];

	}
}

__global__ void increment(double * u, double * v, double * a, double * u_last, double * v_last, int vdim, double dt)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x; // thread id - global number of dof
	if (tid < vdim)
	{
		u[tid] = u_last[tid] + dt*v_last[tid] + 0.5*dt*dt*a[tid];
		v[tid] = v_last[tid] + dt*a[tid];

	}
}