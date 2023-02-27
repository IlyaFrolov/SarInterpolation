#include "interpolation_types.hpp"
#include "interpolation_kernels.hpp"
#include "interpolation.hpp"

#include <iostream>


using interpolation::Matrix;
using Eigen::placeholders::last;
using Eigen::seq;
int main()
{
	Matrix<double> kernels = interpolation::generate_sinc_kernels <double> ();
	Matrix<double> samples = Matrix<double>::Random(2000, 20000);
	Matrix<double> new_indexes = Matrix<double>::Random(2000, 20000);
	Matrix<double> result;

	for (size_t j = 0; j < new_indexes.cols(); j += 1)
		new_indexes(seq(0, last), j) += Matrix<double>::Constant(new_indexes.rows(), 1, j);

	std::cout << "start" << std::endl;
	interpolation::interpolate_cpu(kernels, samples, new_indexes, result);
	std::cout << "end" << std::endl;
	return 0;
}