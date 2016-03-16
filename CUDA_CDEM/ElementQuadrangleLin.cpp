#include "ElementQuadrangleLin.h"

ElementQuadrangleLin::ElementQuadrangleLin(float _E, float _nu, float _density, float _thickness, float _alfaC, int * _nodes, int _nnodes)
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
	M_loc = Eigen::MatrixXf::Identity(stiffness_dim, stiffness_dim);
	M_loc *= volume*density / 4.0;
	M_loc_inv = M_loc.inverse();
	C_loc = M_loc * alfaC;
}

// Calculate local stiffness matrix. Use reduced integration with hourglass stabilization.
void ElementQuadrangleLin::set_K_isoparametric()
{	
	float shear = E / (2 + 2 * nu);
	float lame = E * nu / ((1 + nu) * (1 - 2 * nu));
	Eigen::Matrix3f C;
	C << 2 * shear + lame, lame, 0.,
		lame, 2 * shear + lame, 0.,
		0., 0., shear;
	
	Eigen::Vector4f x, y;
	int i;
	for (i = 0; i < 4; i++)
	{
		x(i) = domain->nodes[nodes[i] - 1].x;
		y(i) = domain->nodes[nodes[i] - 1].y;
	}
	const float gp = sqrt(1. / 3.);
	float gps[5][2] = { { -gp, -gp }, { gp, -gp }, { gp, gp }, { -gp, gp }, { 0., 0. } };
	Eigen::Vector4f g_xi(-0.25, 0.25, 0.25, -0.25);
	Eigen::Vector4f g_eta(-0.25, -0.25, 0.25, 0.25);
	Eigen::Vector4f h(0.25, -0.25, 0.25, -0.25);
	Eigen::Matrix2f J[5];
	Eigen::Matrix2f J_inv[5];
	for (i = 0; i < 5; i++)
	{
		float xi = gps[i][0];
		float eta = gps[i][1];
		J[i](0, 0) = x.dot(g_xi + eta*h);
		J[i](0, 1) = x.dot(g_eta + xi*h);
		J[i](1, 0) = y.dot(g_xi + eta*h);
		J[i](1, 1) = y.dot(g_eta + xi*h);
		J_inv[i] = J[i].inverse();
	}
	Eigen::MatrixXf xieta(4, 2);
	xieta << g_xi, g_eta;
	Eigen::MatrixXf b(4, 2);
	b = xieta*J_inv[4];
	Eigen::MatrixXf B_0T(8,3);
	Eigen::Vector4f zers = Eigen::Vector4f::Zero();
	B_0T << b.block<4,1>(0,0), zers, b.block<4, 1>(0, 1), zers, b.block<4, 1>(0, 1), b.block<4, 1>(0, 0);
	Eigen::MatrixXf xy(4,2);
	xy << x,y;
	Eigen::Matrix4f m_gamma;
	Eigen::Vector4f gamma;
	m_gamma = Eigen::Matrix4f::Identity() - b*xy.transpose();
	gamma = m_gamma * h;
	Eigen::MatrixXf j_0_dev(3, 4);
	Eigen::Matrix2f j0iT = J_inv[4].transpose();
	j_0_dev << 2 * j0iT.block<1, 2>(0, 0), -1 * j0iT.block<1, 2>(1, 0),
		-1 * j0iT.block<1, 2>(0, 0), 2 * j0iT.block<1, 2>(1, 0),
		3 * j0iT.block<1, 2>(0, 0), 3 * j0iT.block<1, 2>(1, 0);
	j_0_dev *= 1. / 3.;
	Eigen::MatrixXf L_hg[4];
	for (i = 0; i < 4; i++)
	{
		float xi = gps[i][0];
		float eta = gps[i][1];
		L_hg[i].resize(4,2);
		L_hg[i] << eta, 0.0, 
					xi, 0.0, 
					0.0, eta, 
					0.0, xi;
	}
	Eigen::MatrixXf M_hg(8,2);
	M_hg << gamma, zers, zers, gamma;
	M_hg.transposeInPlace(); // 2x8
	Eigen::MatrixXf B_red[4];
	Eigen::MatrixXf K_red = Eigen::MatrixXf::Zero(8,8);
	Eigen::MatrixXf B[4];
	Eigen::MatrixXf Bc[4];
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
	Eigen::MatrixXf K(8, 8);
	Eigen::MatrixXf Kc(8, 8);
	Eigen::MatrixXf Kc2(8, 8);
	Kc = K_red + volume*B_0T*C*B_0T.transpose(); // 8x8 + 8x3 * 3x3 * 3x8
	Kc2 << Kc.block<8, 1>(0, 0), Kc.block<8, 1>(0, 4), Kc.block<8, 1>(0, 1), Kc.block<8, 1>(0, 5), Kc.block<8, 1>(0, 2), Kc.block<8, 1>(0, 6), Kc.block<8, 1>(0, 3), Kc.block<8, 1>(0, 7);
	K << Kc2.block<1, 8>(0, 0), Kc2.block<1, 8>(4, 0), Kc2.block<1, 8>(1, 0), Kc2.block<1, 8>(5, 0), Kc2.block<1, 8>(2, 0), Kc2.block<1, 8>(6, 0), Kc2.block<1, 8>(3, 0), Kc2.block<1, 8>(7, 0);
	// Don't forget to swap rows and/or columns of B and K matrix and store it in the object.
	K_loc = K;
	B_matrices = B;
}


// Calculate stress vector for element gauss points.
Eigen::MatrixXf ElementQuadrangleLin::get_stress()
{
	Eigen::MatrixXf sigma;
	return sigma;
}
