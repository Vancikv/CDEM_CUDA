#include <iostream>
#include "Node.h"
#include "aux_functions.h"

Node::Node()
{
}

Node::Node(int _ndofs, float _x, float _y, int * _supports, int * _neighbors, Eigen::VectorXf _v_load)
{
	ndofs = _ndofs;
	x = _x;
	y = _y;
	supports = _supports;
	neighbors = _neighbors;
	v_load = _v_load;
	v_disp = Eigen::VectorXf::Zero(ndofs);
	v_velo = Eigen::VectorXf::Zero(ndofs);
	v_acce = Eigen::VectorXf::Zero(ndofs);
	v_code = new int [ndofs];
	int i;
	for (i = 0; i < ndofs; i++)
	{
		v_code[i] = 0;
	}
}


Node::~Node()
{
	delete[] v_code;
	delete[] supports;
	delete[] neighbors;
}


// Assign code numbers to node dofs. Increase the maxcode value accordingly.
void Node::set_codes(int &maxcode)
{
	int i;
	for (i = 0; i < ndofs; i++)
	{
		if (supports[i] == 0)
		{
			maxcode++;
			v_code[i] = maxcode;
		}
	}
}


// Initiate nodal values prior to a dynamic relaxation calculation.
void Node::init_vals(float tau_0, float mass)
{
	int i;
	for (i = 0; i < ndofs; i++)
	{
		if (supports[i] == 0)
		{
			v_acce(i) = load_function(tau_0) * v_load(i) / mass;
		}
	}
}

// Set the displacement vector.
void Node::set_disp(Eigen::VectorXf val)
{
	v_disp = val;
}

// Set the velocity vector.
void Node::set_velo(Eigen::VectorXf val)
{
	v_velo = val;
}

// Set the acceleration vector.
void Node::set_acce(Eigen::VectorXf val)
{
	v_acce = val;
}
