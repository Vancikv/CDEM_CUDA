#pragma once
#include "Eigen/Dense"
#include "Element.h"
#include "type_manager.h"
#include "type_manager.h"

class ElementQuadrangleLin :
	public Element
{
public:
	ElementQuadrangleLin(FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE, FLOAT_TYPE, int *, int);
	ElementQuadrangleLin();
	~ElementQuadrangleLin();
	// Calculate local stiffness matrix. Use reduced integration with hourglass stabilization.
	void set_K_isoparametric();
	// Calculate stress vector for element gauss points.
	void set_matrices();
	MATRIX_X get_stress();
};

