#pragma once
#include "Element.h"
#include "Node.h"
#include "Eigen/Dense"

class Element;
class Domain
{
public:
	Domain(int, int, Node *, Element *, float, float);
	Domain();
	~Domain();
	Eigen::Matrix2f m_contact_stiffness;
	int nelems, nnodes;
	Element * elements;
	Node * nodes;
	// Create a domain structure defined by a text file
	void load_from_file(std::string filename);
	void write_state_to_file(std::string filename, float time);
	// Calculate the force acting on a node as a result of its relative displacement to the neighbor nodes.
	Eigen::Vector2f get_contact_force(int node_id);
	// Solve the system using the dynamic relaxation method.
	void solve(float t_load, float t_max, int maxiter);
	void solve(float t_load, float t_max, int maxiter, char * outfile, int output_frequency=1, int gridDim=-1, int blockDim=128);
};

