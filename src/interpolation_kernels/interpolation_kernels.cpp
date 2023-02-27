#define _USE_MATH_DEFINES

#include "interpolation_kernels.hpp"
#include "eigen3/Eigen/Dense"
#include <exception>
#include <iostream>


namespace interpolation
{
	using Eigen::seq;
	using Eigen::last;

	template<typename T>
	Matrix<T> generate_sinc_kernels(double step, int stencil_size)
	{
		if (stencil_size % 2 != 0)
			throw std::logic_error("stencil size must be an even number");

		int n_kernels = static_cast<int>(ceil(1.0 / step) + 1.0);
		double real_step = 1.0 / (static_cast<double>(n_kernels) - 1.0);

		Matrix<T> kernels(n_kernels, stencil_size);
		kernels = Matrix<T>::Zero(kernels.rows(), kernels.cols());
		//kernels(seq(0, last), seq(0, last)) = 0.0;

		kernels(0, stencil_size / 2 - 1) = 1;
		kernels(n_kernels - 1, stencil_size / 2) = 1;

		Matrix<T> x_grid(1, stencil_size);
		for (int i = 0; i < x_grid.cols(); i += 1)
			x_grid(0, i) = stencil_size / 2 - 1 - i;


		for (int kernel_i = 1; (kernel_i + 1) < n_kernels; kernel_i += 1)
		{
			double shift = static_cast<double>(kernel_i)*real_step;
			for (int node_i = 0; node_i < stencil_size; node_i += 1)
			{
				double x = x_grid(0, node_i) + shift;
				kernels(kernel_i, node_i) = abs(x) < 1e-6 ? 1.0 : sin(M_PI*x) / (M_PI*x);
			}
			kernels(kernel_i, seq(0, last)) /= kernels(kernel_i, seq(0, last)).norm();
		}

		return kernels;
	}
}

template interpolation::Matrix<double> interpolation::generate_sinc_kernels<double>(double, int);
template interpolation::Matrix<float> interpolation::generate_sinc_kernels<float>(double, int);