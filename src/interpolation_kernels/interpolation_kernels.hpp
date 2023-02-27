#ifndef interpolation_kernels_H
#define interpolation_kernels_H

#include "interpolation_types.hpp"
#include <math.h>
#include "Eigen/Core"
#include "Eigen/Geometry"


namespace interpolation
{
	template<typename T>
	Matrix<T> generate_sinc_kernels(double step=0.1, int stencil_size=8);
}

#endif
