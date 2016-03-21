#include "ElementQuadrangleLin.h"

ElementQuadrangleLin::ElementQuadrangleLin(FLOAT_TYPE _E, FLOAT_TYPE _nu, FLOAT_TYPE _density, FLOAT_TYPE _thickness, FLOAT_TYPE _alfaC, int * _nodes, int _nnodes)
{
	E = _E;
	nu = _nu;
	density = _density;
	thickness = _thickness;
	alfaC = _alfaC;
	nodes = _nodes;
	nnodes = _nnodes;
	stiffness_dim = 8;
}

ElementQuadrangleLin::ElementQuadrangleLin()
{
	stiffness_dim = 8;
}

ElementQuadrangleLin::~ElementQuadrangleLin()
{
	delete[] nodes;
}


// Calculate and store local matrices
void ElementQuadrangleLin::set_matrices()
{
	set_K_isoparametric();
	M_loc = MATRIX_X::Identity(stiffness_dim, stiffness_dim);
	M_loc *= volume*density / 4.0;
	M_loc_inv = M_loc.inverse();
	C_loc = M_loc * alfaC;
}

// Calculate local stiffness matrix. Use reduced integration with hourglass stabilization.
void ElementQuadrangleLin::set_K_isoparametric()
{	
	FLOAT_TYPE shear = E / (2 + 2 * nu);
	FLOAT_TYPE lame = E * nu / ((1 + nu) * (1 - 2 * nu));
	MATRIX_3 C;
	C << 2 * shear + lame, lame, 0.,
		lame, 2 * shear + lame, 0.,
		0., 0., shear;
	
	VECTOR_4 x, y;
	int i;
	for (i = 0; i < 4; i++)
	{
		x(i) = domain->nodes[nodes[i] - 1].x;
		y(i) = domain->nodes[nodes[i] - 1].y;
	}
	const FLOAT_TYPE gp = sqrt(1. / 3.);
	FLOAT_TYPE gps[5][2] = { { -gp, -gp }, { gp, -gp }, { gp, gp }, { -gp, gp }, { 0., 0. } };
	VECTOR_4 g_xi(-0.25, 0.25, 0.25, -0.25);
	VECTOR_4 g_eta(-0.25, -0.25, 0.25, 0.25);
	VECTOR_4 h(0.25, -0.25, 0.25, -0.25);
	MATRIX_2 J[5];
	MATRIX_2 J_inv[5];
	for (i = 0; i < 5; i++)
	{
		FLOAT_TYPE xi = gps[i][0];
		FLOAT_TYPE eta = gps[i][1];
		J[i](0, 0) = x.dot(g_xi + eta*h);
		J[i](0, 1) = x.dot(g_eta + xi*h);
		J[i](1, 0) = y.dot(g_xi + eta*h);
		J[i](1, 1) = y.dot(g_eta + xi*h);
		J_inv[i] = J[i].inverse();
	}
	MATRIX_X xieta(4, 2);
	xieta << g_xi, g_eta;
	MATRIX_X b(4, 2);
	b = xieta*J_inv[4];
	MATRIX_X B_0T(8,3);
	VECTOR_4 zers = VECTOR_4::Zero();
	B_0T << b.block<4,1>(0,0), zers, b.block<4, 1>(0, 1), zers, b.block<4, 1>(0, 1), b.block<4, 1>(0, 0);
	MATRIX_X xy(4,2);
	xy << x,y;
	MATRIX_4 m_gamma;
	VECTOR_4 gamma;
	m_gamma = MATRIX_4::Identity() - b*xy.transpose();
	gamma = m_gamma * h;
	MATRIX_X j_0_dev(3, 4);
	MATRIX_2 j0iT = J_inv[4].transpose();
	j_0_dev << 2 * j0iT.block<1, 2>(0, 0), -1 * j0iT.block<1, 2>(1, 0),
		-1 * j0iT.block<1, 2>(0, 0), 2 * j0iT.block<1, 2>(1, 0),
		3 * j0iT.block<1, 2>(0, 0), 3 * j0iT.block<1, 2>(1, 0);
	j_0_dev *= 1. / 3.;
	MATRIX_X L_hg[4];
	for (i = 0; i < 4; i++)
	{
		FLOAT_TYPE xi = gps[i][0];
		FLOAT_TYPE eta = gps[i][1];
		L_hg[i].resize(4,2);
		L_hg[i] << eta, 0.0, 
					xi, 0.0, 
					0.0, eta, 
					0.0, xi;
	}
	MATRIX_X M_hg(8,2);
	M_hg << gamma, zers, zers, gamma;
	M_hg.transposeInPlace(); // 2x8
	MATRIX_X B_red[4];
	MATRIX_X K_red = MATRIX_X::Zero(8,8);
	MATRIX_X B[4];
	MATRIX_X Bc[4];
	volume = 4 * J[4].determinant()*thickness;
	for (i = 0; i < 4; i++)
	{
		B_red[i].resize(3, 8);
		B[i].resize(3, 8);
		Bc[i].resize(3, 8);
		B_red[i] = j_0_dev * L_hg[i] * M_hg; // 3x4 * 4x2 * 2x8 = 3x8
		K_red += B_red[i].transpose()*C*B_red[i]; // 8x3 * 3x3 * 3x8 = 8x8
		Bc[i] = B_0T.transpose() + B_red[i]; // 3x8 + 3x8 = 3x8
		B[i] << Bc[i].block<3, 1>(0, 0), Bc[i].block<3, 1>(0, 4), Bc[i].block<3, 1>(0, 1), Bc[i].block<3, 1>(0, 5),	Bc[i].block<3, 1>(0, 2), Bc[i].block<3, 1>(0, 6), Bc[i].block<3, 1>(0, 3), Bc[i].block<3, 1>(0, 7); // 3x8
	}
	K_red *= volume / 4.0;
	MATRIX_X K(8, 8);
	MATRIX_X Kc(8, 8);
	MATRIX_X Kc2(8, 8);
	Kc = K_red + volume*B_0T*C*B_0T.transpose(); // 8x8 + 8x3 * 3x3 * 3x8
	Kc2 << Kc.block<8, 1>(0, 0), Kc.block<8, 1>(0, 4), Kc.block<8, 1>(0, 1), Kc.block<8, 1>(0, 5), Kc.block<8, 1>(0, 2), Kc.block<8, 1>(0, 6), Kc.block<8, 1>(0, 3), Kc.block<8, 1>(0, 7);
	K << Kc2.block<1, 8>(0, 0), Kc2.block<1, 8>(4, 0), Kc2.block<1, 8>(1, 0), Kc2.block<1, 8>(5, 0), Kc2.block<1, 8>(2, 0), Kc2.block<1, 8>(6, 0), Kc2.block<1, 8>(3, 0), Kc2.block<1, 8>(7, 0);
	// Don't forget to swap rows and/or columns of B and K matrix and store it in the object.
	K_loc = K;
	B_matrices = B;
}


// Calculate stress vector for element gauss points.
MATRIX_X ElementQuadrangleLin::get_stress()
{
	MATRIX_X sigma;
	return sigma;
}
