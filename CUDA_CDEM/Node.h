#pragma once
#include "Eigen/Dense"

class Node
{
public:
	Node();
	Node(int, float, float, int *, int *, Eigen::VectorXf);
	~Node();
	int ndofs;
	float x;
	float y;
	Eigen::VectorXf v_disp;
	Eigen::VectorXf v_velo;
	Eigen::VectorXf v_acce;
	int * v_code;
	Eigen::Vector2f v_norm[2];
	Eigen::VectorXf v_load;
	int * supports;
	int * neighbors;
	// Assign code numbers to node dofs. Increase the maxcode value accordingly.
	void set_codes(int &maxcode);
	// Initiate nodal values prior to a dynamic relaxation calculation.
	void init_vals(float tau_0, float mass);
	// Set the displacement vector.
	void set_disp(Eigen::VectorXf);
	// Set the velocity vector.
	void set_velo(Eigen::VectorXf);
	// Set the acceleration vector.
	void set_acce(Eigen::VectorXf);
};

