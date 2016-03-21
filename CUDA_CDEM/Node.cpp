#include <iostream>
#include "Node.h"
#include "aux_functions.h"

Node::Node()
{
}

Node::Node(int _ndofs, FLOAT_TYPE _x, FLOAT_TYPE _y, int * _supports, int * _neighbors, VECTOR_X _v_load)
{
	ndofs = _ndofs;
	x = _x;
	y = _y;
	supports = _supports;
	neighbors = _neighbors;
	v_load = _v_load;
	v_disp = VECTOR_X::Zero(ndofs);
	v_velo = VECTOR_X::Zero(ndofs);
	v_acce = VECTOR_X::Zero(ndofs);
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
void Node::init_vals(FLOAT_TYPE tau_0, FLOAT_TYPE mass)
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
void Node::set_disp(VECTOR_X val)
{
	v_disp = val;
}

// Set the velocity vector.
void Node::set_velo(VECTOR_X val)
{
	v_velo = val;
}

// Set the acceleration vector.
void Node::set_acce(VECTOR_X val)
{
	v_acce = val;
}
