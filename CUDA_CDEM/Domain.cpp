#include <iostream>
#include "Domain.h"
#include "ElementQuadrangleLin.h"
#include "Node.h"
#include <fstream>
#include <istream>
#include "cuda_functions.cuh"
#include "aux_functions.h"
#include "time_measuring.h"

Domain::Domain(int _nelems, int _nnodes, Node * _nodes, Element * _elements, FLOAT_TYPE c_n, FLOAT_TYPE c_s)
{
	nelems = _nelems;
	nnodes = _nnodes;
	nodes = _nodes;
	elements = _elements;
	m_contact_stiffness << c_n, 0.0,
		0.0, c_s;
}

Domain::Domain()
{
	m_contact_stiffness << 0.0, 0.0,
		0.0, 0.0;
}

Domain::~Domain()
{
	delete[] elements;
	delete[] nodes;
}

void Domain::write_state_to_file(std::string filename, FLOAT_TYPE time)
{
	std::ofstream outfile(filename);
	int i;

	outfile << nnodes << " " << nelems << " " << time << std::endl;
	for (i = 0; i < nnodes; i++)
	{
		outfile << "node " << (i+1) << " " << nodes[i].x << " " << nodes[i].y << " " << nodes[i].v_disp(0) << " " << nodes[i].v_disp(1)
			<< " " << nodes[i].v_velo(0) << " " << nodes[i].v_velo(0) << " " << nodes[i].v_acce(0) << " " << nodes[i].v_acce(0) << std::endl;
	}
	for (i = 0; i < nelems; i++)
	{
		Element& e = elements[i];
		outfile << "element " << (i + 1) << " " << elements[i].nnodes;
		for (int j = 0; j < elements[i].nnodes; j++)
		{
			outfile << " " << elements[i].nodes[j];
		}
		outfile << std::endl;
	}
}

// Calculate the force acting on a node as a result of its relative displacement to the neighbor nodes.
VECTOR_2 Domain::get_contact_force(int node_id)
{
	VECTOR_2 F(0., 0.);
	VECTOR_2 du_g;
	MATRIX_2 T;
	Node& n0 = nodes[node_id-1];
	int i;
	for (i = 0; i < 2; i++)
	{
		if (n0.neighbors[i] != 0)
		{
			T << n0.v_norm[i](0), n0.v_norm[i](1),
				-n0.v_norm[i](1), n0.v_norm[i](0);
			du_g = nodes[n0.neighbors[i] - 1].v_disp - n0.v_disp; // 2x1 - 2x1 = 2x1
			F += T.transpose() * m_contact_stiffness * T * du_g; // 2x2 * 2x2 * 2x2 * 2x1 = 2x1
		}
	}
	return F;
}

// Solve the system using the dynamic relaxation method.
void Domain::solve(FLOAT_TYPE t_load, FLOAT_TYPE t_max, int maxiter)
{	
	FLOAT_TYPE dt = t_max / maxiter;
	int i, j;
	for (i = 0; i < nelems; i++)
	{
		elements[i].set_matrices();
		elements[i].calc_normal_vectors();
		for (j = 0; j < elements[i].nnodes; j++)
		{
			nodes[elements[i].nodes[j]-1].init_vals(dt/t_max, elements[i].M_loc(2 * j, 2 * j));
		}
	}

	// Eigen matrices will be copied into arrays of FLOAT_TYPEs.
	// Using the Eigen::Map function defaults in a column by column layout.
	FLOAT_TYPE * u, *v, *a, *load, *supports, *K, *C, *Mi, *Kc, *n_vects;
	int * neighbors;
	int nnodedofs = 2, stiffdim = 8;
	int vdim = nnodes*nnodedofs;
	u = new FLOAT_TYPE[vdim];
	v = new FLOAT_TYPE[vdim];
	a = new FLOAT_TYPE[vdim];
	load = new FLOAT_TYPE[vdim];
	supports = new FLOAT_TYPE[vdim];
	K = new FLOAT_TYPE[nelems*stiffdim*stiffdim];
	C = new FLOAT_TYPE[nelems*stiffdim];
	Mi = new FLOAT_TYPE[nelems*stiffdim];
	Kc = new FLOAT_TYPE[4];
	n_vects = new FLOAT_TYPE[4 * nnodes];
	neighbors = new int[vdim];

	for (i = 0; i < nnodes; i++)
	{
		int of = i*nnodedofs;
		Eigen::Map<VECTOR_X>(u + of, nodes[i].v_disp.rows(), nodes[i].v_disp.cols()) = nodes[i].v_disp;
		Eigen::Map<VECTOR_X>(v + of, nodes[i].v_velo.rows(), nodes[i].v_velo.cols()) = nodes[i].v_velo;
		Eigen::Map<VECTOR_X>(a + of, nodes[i].v_acce.rows(), nodes[i].v_acce.cols()) = nodes[i].v_acce;
		Eigen::Map<VECTOR_X>(load + of, nodes[i].v_load.rows(), nodes[i].v_load.cols()) = nodes[i].v_load;
		*(supports++) = nodes[i].supports[0];
		*(supports++) = nodes[i].supports[1];
		*(neighbors++) = nodes[i].neighbors[0];
		*(neighbors++) = nodes[i].neighbors[1];
		*(n_vects++) = nodes[i].v_norm[0](0);
		*(n_vects++) = nodes[i].v_norm[0](1);
		*(n_vects++) = nodes[i].v_norm[1](0);
		*(n_vects++) = nodes[i].v_norm[1](1);

	}

	supports -= vdim;
	neighbors -= vdim;
	n_vects -= 4 * nnodes;

	for (i = 0; i < nelems; i++)
	{
		int of1 = i*stiffdim*stiffdim, of2 = i*stiffdim;
		Eigen::Map<MATRIX_X>(K + of1, elements[i].K_loc.rows(), elements[i].K_loc.cols()) = elements[i].K_loc;
		Eigen::Map<MATRIX_X>(C + of2, elements[i].C_loc.diagonal().rows(), elements[i].C_loc.diagonal().cols()) = elements[i].C_loc.diagonal();
		Eigen::Map<MATRIX_X>(Mi + of2, elements[i].M_loc_inv.diagonal().rows(), elements[i].M_loc_inv.diagonal().cols()) = elements[i].M_loc_inv.diagonal();
	}

	Eigen::Map<MATRIX_X>(Kc, m_contact_stiffness.rows(), m_contact_stiffness.cols()) = m_contact_stiffness;

	//element_step_with_CUDA(u, v, a, load, supports, neighbors, n_vects, K, C, Mi, Kc, nelems, nnodes, nnodedofs, 
	//		stiffdim,t_load,t_max,maxiter);
	std::cout << "Deflection of the 59th node: " << u[117] << std::endl;
		//for (j = 0; j < nelems; j++)
		//{
		//	elements[j].iterate(dt, i * dt / t_load, true);
		//}
}

void Domain::solve(FLOAT_TYPE t_load, FLOAT_TYPE t_max, int maxiter, char* outfile, int output_frequency, int gridDim, int blockDim)
{
	FLOAT_TYPE dt = t_max / maxiter;
	int i, j;
	FLOAT_TYPE solve_begin_CPU = get_cpu_time();
	FLOAT_TYPE solve_begin_wall = get_wall_time();
	for (i = 0; i < nelems; i++)
	{
		elements[i].set_matrices();
		elements[i].calc_normal_vectors();
		for (j = 0; j < elements[i].nnodes; j++)
		{
			nodes[elements[i].nodes[j] - 1].init_vals(dt / t_max, elements[i].M_loc(2 * j, 2 * j));
		}
	}

	// Eigen matrices will be copied into arrays of FLOAT_TYPEs.
	// Using the Eigen::Map function defaults in a column by column layout.
	FLOAT_TYPE * u, *v, *a, *load, *supports, *K, *C, *Mi, *Kc, *n_vects;
	int * neighbors;
	int nnodedofs = 2, stiffdim = 8;
	int vdim = nnodes*nnodedofs;
	u = new FLOAT_TYPE[vdim];
	v = new FLOAT_TYPE[vdim];
	a = new FLOAT_TYPE[vdim];
	load = new FLOAT_TYPE[vdim];
	supports = new FLOAT_TYPE[vdim];
	K = new FLOAT_TYPE[nelems*stiffdim*stiffdim];
	C = new FLOAT_TYPE[nelems*stiffdim];
	Mi = new FLOAT_TYPE[nelems*stiffdim];
	Kc = new FLOAT_TYPE[4];
	n_vects = new FLOAT_TYPE[4 * nnodes];
	neighbors = new int[vdim];

	for (i = 0; i < nnodes; i++)
	{
		int of = i*nnodedofs;
		Eigen::Map<VECTOR_X>(u + of, nodes[i].v_disp.rows(), nodes[i].v_disp.cols()) = nodes[i].v_disp;
		Eigen::Map<VECTOR_X>(v + of, nodes[i].v_velo.rows(), nodes[i].v_velo.cols()) = nodes[i].v_velo;
		Eigen::Map<VECTOR_X>(a + of, nodes[i].v_acce.rows(), nodes[i].v_acce.cols()) = nodes[i].v_acce;
		Eigen::Map<VECTOR_X>(load + of, nodes[i].v_load.rows(), nodes[i].v_load.cols()) = nodes[i].v_load;
		*(supports++) = nodes[i].supports[0];
		*(supports++) = nodes[i].supports[1];
		*(neighbors++) = nodes[i].neighbors[0];
		*(neighbors++) = nodes[i].neighbors[1];
		*(n_vects++) = nodes[i].v_norm[0](0);
		*(n_vects++) = nodes[i].v_norm[0](1);
		*(n_vects++) = nodes[i].v_norm[1](0);
		*(n_vects++) = nodes[i].v_norm[1](1);
	}

	supports -= vdim;
	neighbors -= vdim;
	n_vects -= 4 * nnodes;

	for (i = 0; i < nelems; i++)
	{
		int of1 = i*stiffdim*stiffdim, of2 = i*stiffdim;
		Eigen::Map<MATRIX_X>(K + of1, elements[i].K_loc.rows(), elements[i].K_loc.cols()) = elements[i].K_loc;
		Eigen::Map<MATRIX_X>(C + of2, elements[i].C_loc.diagonal().rows(), elements[i].C_loc.diagonal().cols()) = elements[i].C_loc.diagonal();
		Eigen::Map<MATRIX_X>(Mi + of2, elements[i].M_loc_inv.diagonal().rows(), elements[i].M_loc_inv.diagonal().cols()) = elements[i].M_loc_inv.diagonal();
	}

	Eigen::Map<MATRIX_X>(Kc, m_contact_stiffness.rows(), m_contact_stiffness.cols()) = m_contact_stiffness;
	std::string num = "00.txt", fl=outfile;
	write_state_to_file(fl + num, 0);
	element_step_with_CUDA(u, v, a, load, supports, neighbors, n_vects, K, C, Mi, Kc, nelems, nnodes, nnodedofs,
		stiffdim, t_load, t_max, maxiter, outfile, output_frequency, gridDim, blockDim);
	FLOAT_TYPE solve_end_CPU = get_cpu_time();
	FLOAT_TYPE solve_end_wall = get_wall_time();
	std::cout << "CPU time taken: " << solve_end_CPU - solve_begin_CPU << " s" << std::endl;
	std::cout << "Wall time taken: " << solve_end_wall - solve_begin_wall << " s" << std::endl;
	//for (j = 0; j < nelems; j++)
	//{
	//	elements[j].iterate(dt, i * dt / t_load, true);
	//}
}

void Domain::load_from_file(std::string filename)
{
	std::ifstream input_file (filename);
	std::string line;
	int lncount=0;

	while (std::getline(input_file, line)) // Implement error catching!
	{
		lncount++;
		std::string entry;
		std::stringstream lss(line);
		if (lncount == 1) // First row - domain specs
		{
			while (std::getline(lss,entry,' '))
			{
				if (entry == "nnodes") lss >> nnodes;
				if (entry == "nelems") lss >> nelems;
				if (entry == "cn") lss >> m_contact_stiffness(0,0);
				if (entry == "cs") lss >> m_contact_stiffness(1,1);
			}
			elements = new ElementQuadrangleLin[nelems];
			nodes = new Node[nnodes];
		}
		else if (lncount <= 1 + nnodes) // Node records
		{
			Node& nd = nodes[lncount - 2];
			while (std::getline(lss, entry, ' '))
			{
				if (entry == "ndofs")
				{
					int ndf, i;
					lss >> ndf;
					nd.ndofs = ndf;
					nd.neighbors = new int[ndf];
					nd.supports = new int[ndf];
					nd.v_load = VECTOR_X::Zero(ndf);
					nd.v_disp = VECTOR_X::Zero(ndf);
					nd.v_velo = VECTOR_X::Zero(ndf);
					nd.v_acce = VECTOR_X::Zero(ndf);
					nd.v_code = new int[ndf];
					for (i = 0; i < ndf; i++)
					{
						nd.neighbors[i] = 0;
						nd.supports[i] = 0;
						nd.v_code[i] = 0;
					}
				}
				if (entry == "position")
				{
					lss >> nd.x;
					lss >> nd.y;
				}
				if (entry == "neighbors")
				{
					lss >> nd.neighbors[0];
					lss >> nd.neighbors[1];
				}
				if (entry == "supports")
				{
					int i;
					for (i = 0; i < nd.ndofs;i++) lss >> nd.supports[i];
				}
				if (entry == "load")
				{
					int i;
					for (i = 0; i < nd.ndofs; i++) lss >> nd.v_load(i);
				}
			}
		}
		else // Element records
		{
			Element& el = elements[lncount - 2 - nnodes];
			el.domain = this;
			while (std::getline(lss, entry, ' '))
			{
				if (entry == "nodes")
				{
					int nnds, i;
					lss >> nnds;
					el.nnodes = nnds;
					el.nodes = new int[nnds];
					for (i = 0; i < nnds; i++)
					{
						lss >> el.nodes[i];
					}
				}
				if (entry == "E") lss >> el.E;
				if (entry == "nu") lss >> el.nu;
				if (entry == "density") lss >> el.density;
				if (entry == "thickness") lss >> el.thickness;
				if (entry == "alfaC") lss >> el.alfaC;
			}
		}
	}

}