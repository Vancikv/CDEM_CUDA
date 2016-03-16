#pragma once
#include "Eigen/Dense"
#include "Element.h"

class ElementQuadrangleLin :
	public Element
{
public:
	ElementQuadrangleLin(float, float, float, float, float, int *, int);
	ElementQuadrangleLin();
	~ElementQuadrangleLin();
	// Calculate local stiffness matrix. Use reduced integration with hourglass stabilization.
	void set_K_isoparametric();
	// Calculate stress vector for element gauss points.
	void set_matrices();
	Eigen::MatrixXf get_stress();
};

