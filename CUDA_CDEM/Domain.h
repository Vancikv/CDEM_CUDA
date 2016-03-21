#pragma once
#include "Element.h"
#include "Node.h"
#include "type_manager.h"
#include "Eigen/Dense"

class Element;
class Domain
{
public:
	Domain(int, int, Node *, Element *, FLOAT_TYPE, FLOAT_TYPE);
	Domain();
	~Domain();
	MATRIX_2 m_contact_stiffness;
	int nelems, nnodes;
	Element * elements;
	Node * nodes;
	// Create a domain structure defined by a text file
	void load_from_file(std::string filename);
	void write_state_to_file(std::string filename, FLOAT_TYPE time);
	// Calculate the force acting on a node as a result of its relative displacement to the neighbor nodes.
	VECTOR_2 get_contact_force(int node_id);
	// Solve the system using the dynamic relaxation method.
	void solve(FLOAT_TYPE t_load, FLOAT_TYPE t_max, int maxiter);
	void solve(FLOAT_TYPE t_load, FLOAT_TYPE t_max, int maxiter, char * outfile, int output_frequency=1, int gridDim=-1, int blockDim=128);
};

