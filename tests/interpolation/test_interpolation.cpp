#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include <iostream>
#include "interpolation_types.hpp"
#include "interpolation.hpp"

template<typename T>
T interpolate(const interpolation::Matrix<T>& samples, const interpolation::Matrix<T>& kernels, int line_i, double new_index)
{
	int n_kernels = kernels.rows();
	int stencil_size = kernels.cols();

	int N = samples.cols();

	double step = 1.0 / (n_kernels - 1.0);

	int sample_i = static_cast<int>(floor(new_index));
	double shift = new_index - sample_i;

	int kernel_i = static_cast<int>(round(shift / step));

	int sample1 = sample_i - stencil_size / 2 + 1, sample2 = sample_i + stencil_size / 2;
	int kernel_sample1 = 0, kernel_sample2 = stencil_size - 1;

	if ((sample_i + stencil_size / 2) < 0) return 0;
	if ((sample_i - stencil_size / 2 + 1) > (N - 1)) return 0;
	if ((sample_i - stencil_size / 2 + 1) < 0)
	{
		sample1 = 0;
		kernel_sample1 = -(sample_i - stencil_size / 2 + 1);
	}
	if ((sample_i + stencil_size / 2) > (N - 1))
	{
		sample2 = N - 1;
		kernel_sample2 = stencil_size - 1 - ((sample_i + stencil_size / 2) - (N - 1));
	}

	assert(kernel_sample1 <= kernel_sample2);
	assert((sample2 - sample1) == (kernel_sample2 - kernel_sample1));
	T res = 0;
	for (int i = 0; i <= (sample2 - sample1); i += 1)
		res += samples(line_i, sample1 + i)*kernels(kernel_i, kernel_sample1 + i);

	return res;
}


// test implementation for float and double template type
TEMPLATE_TEST_CASE("test interpolation", "[]", float, double)
{
	int stencil_size = 4;
	float step = 0.1;

	interpolation::Matrix<TestType> kernels = interpolation::generate_sinc_kernels<TestType>(step, stencil_size);

	interpolation::Matrix<TestType> samples(2, 8);
	samples << 1, 2, 3, 4, 5, 6, 7, 8,
		       2, 4, 6, 8, 10, 12, 14, 16;

	interpolation::Matrix<double> new_indexes(2, 6);
	new_indexes << -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 
		            4.5, 5.5, 6.5, 7.5, 8.5, 9.5; 
		        

	interpolation::Matrix<TestType> result_ideal(new_indexes.rows(), new_indexes.cols());

	for (int i = 0; i < result_ideal.rows(); i += 1)
		for (int j = 0; j < result_ideal.cols(); j += 1)
			result_ideal(i, j) = interpolate<TestType>(samples, kernels, i, new_indexes(i, j));

	interpolation::Matrix<TestType> result;
	interpolation::interpolate_cpu(kernels, samples, new_indexes, result);

	// check if sizes are equal
	REQUIRE(result_ideal.rows() == result.rows());
	REQUIRE(result_ideal.cols() == result.cols());

	//std::cout << result_ideal << std::endl;
	//std::cout << std::endl;
	//std::cout << result << std::endl;


	// check error
	interpolation::Matrix<TestType> err = result_ideal - result;
	TestType min_coeff = err.minCoeff();
	TestType max_coeff = err.maxCoeff();
	TestType max_err = std::max(std::abs(min_coeff), std::abs(max_coeff));

	REQUIRE(max_err < 1e-6);
}