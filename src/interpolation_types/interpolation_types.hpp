#ifndef interpolation_types_H
#define interpolation_types_H

#include <eigen3/Eigen/Dense>

namespace interpolation
{
	template<typename T>
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
}

#endif