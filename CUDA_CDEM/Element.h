#pragma once
#include "Domain.h"
#include "Eigen/Dense"

class Domain;
class Element
{
public:
	Element();
	virtual ~Element();
	float E;
	float nu;
	float density;
	float thickness;
	float alfaC;
	Domain * domain;
	int nnodes;
	int * nodes;
	int stiffness_dim;
	Eigen::MatrixXf K_loc;
	Eigen::MatrixXf * B_matrices;
	Eigen::MatrixXf M_loc;
	Eigen::MatrixXf M_loc_inv;
	Eigen::MatrixXf C_loc;
	float volume;
	// print data
	void print_self();
	// Calculate and store local matrices
	virtual void set_matrices();
	// Calculate the normal vector of each face and pass it to the adjacent nodes.
	void calc_normal_vectors();
	// Perform one iteration of dynamic relaxation. Return the velocity norm.
	float iterate(float dt, float tau, bool verbose=false);
};
