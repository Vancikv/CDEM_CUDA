#pragma once
#include "Eigen/Dense"
#include "type_manager.h"

class Node
{
public:
	Node();
	Node(int, FLOAT_TYPE, FLOAT_TYPE, int *, int *, VECTOR_X);
	~Node();
	int ndofs;
	FLOAT_TYPE x;
	FLOAT_TYPE y;
	VECTOR_X v_disp;
	VECTOR_X v_velo;
	VECTOR_X v_acce;
	int * v_code;
	VECTOR_2 v_norm[2];
	VECTOR_X v_load;
	int * supports;
	int * neighbors;
	// Assign code numbers to node dofs. Increase the maxcode value accordingly.
	void set_codes(int &maxcode);
	// Initiate nodal values prior to a dynamic relaxation calculation.
	void init_vals(FLOAT_TYPE tau_0, FLOAT_TYPE mass);
	// Set the displacement vector.
	void set_disp(VECTOR_X);
	// Set the velocity vector.
	void set_velo(VECTOR_X);
	// Set the acceleration vector.
	void set_acce(VECTOR_X);
};

