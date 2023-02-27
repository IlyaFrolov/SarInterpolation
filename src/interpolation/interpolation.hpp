#ifndef interpolation_H
#define interpolation_H

#include "interpolation_types.hpp"
#include <math.h>
#include "interpolation_kernels.hpp"

namespace interpolation
{
	template<typename T>
	void interpolate_cpu(const Matrix<T>& kernels, const Matrix<T>& samples, const Matrix<double>& new_indexes, Matrix<T>& result);
}

#endif