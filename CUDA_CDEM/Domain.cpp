#include <iostream>
#include "Domain.h"
#include "ElementQuadrangleLin.h"
#include "Node.h"
#include <fstream>
#include <istream>
#include "cuda_functions.cuh"
#include "aux_functions.h"

Domain::Domain(int _nelems, int _nnodes, Node * _nodes, Element * _elements, double c_n, double c_s)
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

void Domain::write_state_to_file(std::string filename, double time)
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
Eigen::Vector2d Domain::get_contact_force(int node_id)
{
	Eigen::Vector2d F(0., 0.);
	Eigen::Vector2d du_g;
	Eigen::Matrix2d T;
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
void Domain::solve(double t_load, double t_max, int maxiter)
{	
	float dt = t_max / maxiter;
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

	// Eigen matrices will be copied into arrays of doubles.
	// Using the Eigen::Map function defaults in a column by column layout.
	double * u, *v, *a, *load, *supports, *K, *C, *Mi, *Kc, *n_vects;
	int * neighbors;
	int nnodedofs = 2, stiffdim = 8;
	int vdim = nnodes*nnodedofs;
	u = new double[vdim];
	v = new double[vdim];
	a = new double[vdim];
	load = new double[vdim];
	supports = new double[vdim];
	K = new double[nelems*stiffdim*stiffdim];
	C = new double[nelems*stiffdim];
	Mi = new double[nelems*stiffdim];
	Kc = new double[4];
	n_vects = new double[4 * nnodes];
	neighbors = new int[vdim];

	for (i = 0; i < nnodes; i++)
	{
		int of = i*nnodedofs;
		Eigen::Map<Eigen::VectorXd>(u + of, nodes[i].v_disp.rows(), nodes[i].v_disp.cols()) = nodes[i].v_disp;
		Eigen::Map<Eigen::VectorXd>(v + of, nodes[i].v_velo.rows(), nodes[i].v_velo.cols()) = nodes[i].v_velo;
		Eigen::Map<Eigen::VectorXd>(a + of, nodes[i].v_acce.rows(), nodes[i].v_acce.cols()) = nodes[i].v_acce;
		Eigen::Map<Eigen::VectorXd>(load + of, nodes[i].v_load.rows(), nodes[i].v_load.cols()) = nodes[i].v_load;
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
		Eigen::Map<Eigen::MatrixXd>(K + of1, elements[i].K_loc.rows(), elements[i].K_loc.cols()) = elements[i].K_loc;
		Eigen::Map<Eigen::MatrixXd>(C + of2, elements[i].C_loc.diagonal().rows(), elements[i].C_loc.diagonal().cols()) = elements[i].C_loc.diagonal();
		Eigen::Map<Eigen::MatrixXd>(Mi + of2, elements[i].M_loc_inv.diagonal().rows(), elements[i].M_loc_inv.diagonal().cols()) = elements[i].M_loc_inv.diagonal();
	}

	Eigen::Map<Eigen::MatrixXd>(Kc, m_contact_stiffness.rows(), m_contact_stiffness.cols()) = m_contact_stiffness;

	//element_step_with_CUDA(u, v, a, load, supports, neighbors, n_vects, K, C, Mi, Kc, nelems, nnodes, nnodedofs, 
	//		stiffdim,t_load,t_max,maxiter);
	std::cout << "Deflection of the 59th node: " << u[117] << std::endl;
		//for (j = 0; j < nelems; j++)
		//{
		//	elements[j].iterate(dt, i * dt / t_load, true);
		//}
}

void Domain::solve(double t_load, double t_max, int maxiter, char* outfile, int output_frequency, int gridDim, int blockDim)
{
	float dt = t_max / maxiter;
	int i, j;
	for (i = 0; i < nelems; i++)
	{
		elements[i].set_matrices();
		elements[i].calc_normal_vectors();
		for (j = 0; j < elements[i].nnodes; j++)
		{
			nodes[elements[i].nodes[j] - 1].init_vals(dt / t_max, elements[i].M_loc(2 * j, 2 * j));
		}
	}

	// Eigen matrices will be copied into arrays of doubles.
	// Using the Eigen::Map function defaults in a column by column layout.
	double * u, *v, *a, *load, *supports, *K, *C, *Mi, *Kc, *n_vects;
	int * neighbors;
	int nnodedofs = 2, stiffdim = 8;
	int vdim = nnodes*nnodedofs;
	u = new double[vdim];
	v = new double[vdim];
	a = new double[vdim];
	load = new double[vdim];
	supports = new double[vdim];
	K = new double[nelems*stiffdim*stiffdim];
	C = new double[nelems*stiffdim];
	Mi = new double[nelems*stiffdim];
	Kc = new double[4];
	n_vects = new double[4 * nnodes];
	neighbors = new int[vdim];

	for (i = 0; i < nnodes; i++)
	{
		int of = i*nnodedofs;
		Eigen::Map<Eigen::VectorXd>(u + of, nodes[i].v_disp.rows(), nodes[i].v_disp.cols()) = nodes[i].v_disp;
		Eigen::Map<Eigen::VectorXd>(v + of, nodes[i].v_velo.rows(), nodes[i].v_velo.cols()) = nodes[i].v_velo;
		Eigen::Map<Eigen::VectorXd>(a + of, nodes[i].v_acce.rows(), nodes[i].v_acce.cols()) = nodes[i].v_acce;
		Eigen::Map<Eigen::VectorXd>(load + of, nodes[i].v_load.rows(), nodes[i].v_load.cols()) = nodes[i].v_load;
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
		Eigen::Map<Eigen::MatrixXd>(K + of1, elements[i].K_loc.rows(), elements[i].K_loc.cols()) = elements[i].K_loc;
		Eigen::Map<Eigen::MatrixXd>(C + of2, elements[i].C_loc.diagonal().rows(), elements[i].C_loc.diagonal().cols()) = elements[i].C_loc.diagonal();
		Eigen::Map<Eigen::MatrixXd>(Mi + of2, elements[i].M_loc_inv.diagonal().rows(), elements[i].M_loc_inv.diagonal().cols()) = elements[i].M_loc_inv.diagonal();
	}

	Eigen::Map<Eigen::MatrixXd>(Kc, m_contact_stiffness.rows(), m_contact_stiffness.cols()) = m_contact_stiffness;
	std::string num = "00.txt", fl=outfile;
	write_state_to_file(fl + num, 0);
	element_step_with_CUDA(u, v, a, load, supports, neighbors, n_vects, K, C, Mi, Kc, nelems, nnodes, nnodedofs,
		stiffdim, t_load, t_max, maxiter, outfile, output_frequency, gridDim, blockDim);
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
					nd.v_load = Eigen::VectorXd::Zero(ndf);
					nd.v_disp = Eigen::VectorXd::Zero(ndf);
					nd.v_velo = Eigen::VectorXd::Zero(ndf);
					nd.v_acce = Eigen::VectorXd::Zero(ndf);
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