#pragma once
#include "Domain.h"
#include "type_manager.h"
#include "Eigen/Dense"

class Domain;
class Element
{
public:
	Element();
	virtual ~Element();
	FLOAT_TYPE E;
	FLOAT_TYPE nu;
	FLOAT_TYPE density;
	FLOAT_TYPE thickness;
	FLOAT_TYPE alfaC;
	Domain * domain;
	int nnodes;
	int * nodes;
	int stiffness_dim;
	MATRIX_X K_loc;
	MATRIX_X * B_matrices;
	MATRIX_X M_loc;
	MATRIX_X M_loc_inv;
	MATRIX_X C_loc;
	FLOAT_TYPE volume;
	// print data
	void print_self();
	// Calculate and store local matrices
	virtual void set_matrices();
	// Calculate the normal vector of each face and pass it to the adjacent nodes.
	void calc_normal_vectors();
	// Perform one iteration of dynamic relaxation. Return the velocity norm.
	FLOAT_TYPE iterate(FLOAT_TYPE dt, FLOAT_TYPE tau, bool verbose=false);
};
