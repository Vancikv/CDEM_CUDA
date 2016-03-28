#pragma once

#define FLOAT_TYPE double
// Toggle the type used for most floating point variables. Mostly switch float/double.
#define EIGEN_SUFFIX d
// Toggle the type for matrix elements, float/double/...

#define CONCAT(a,b) a##b
// Concatenate two strings, macros NOT expanded first.
#define CONCAT_EX(a,b) CONCAT(a,b)
// Expand a and b if they are macros, concatenate after.

#define VECTOR_2 CONCAT_EX(Eigen::Vector2,EIGEN_SUFFIX)
#define VECTOR_3 CONCAT_EX(Eigen::Vector3,EIGEN_SUFFIX)
#define VECTOR_4 CONCAT_EX(Eigen::Vector4,EIGEN_SUFFIX)
#define VECTOR_X CONCAT_EX(Eigen::VectorX,EIGEN_SUFFIX)

#define MATRIX_2 CONCAT_EX(Eigen::Matrix2,EIGEN_SUFFIX)
#define MATRIX_3 CONCAT_EX(Eigen::Matrix3,EIGEN_SUFFIX)
#define MATRIX_4 CONCAT_EX(Eigen::Matrix4,EIGEN_SUFFIX)
#define MATRIX_X CONCAT_EX(Eigen::MatrixX,EIGEN_SUFFIX)
