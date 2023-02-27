#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include <iostream>
#include "interpolation_types.hpp"
#include "interpolation_kernels.hpp"

// test implementation for float and double template type
TEMPLATE_TEST_CASE("test sinc kernels", "[]", float, double)
{
	int stencil_size = 4;
	float step = 0.3;



	Eigen::Matrix<TestType, 5, 4> ideal_kernels;

	ideal_kernels << 0.00000000, 1.00000000, 0.00000000, 0.00000000,
		             -0.18478013, 0.92390067, 0.30796689, -0.13198581,
		             -0.22360680, 0.67082039, 0.67082039, -0.22360680,
		             -0.13198581, 0.30796689, 0.92390067, -0.18478013,
		             0.00000000, 0.00000000, 1.00000000, 0.00000000;

	interpolation::Matrix<TestType> kernels = interpolation::generate_sinc_kernels<TestType>(step, stencil_size);

	// check if sizes are equal
	REQUIRE(ideal_kernels.rows() == kernels.rows());
	REQUIRE(ideal_kernels.cols() == kernels.cols());
	
	// check error
	Eigen::Matrix<TestType, 5, 4> err = ideal_kernels - kernels;
	TestType min_coeff = err.minCoeff();
	TestType max_coeff = err.maxCoeff();
	TestType max_err = std::max(std::abs(min_coeff), std::abs(max_coeff));

	REQUIRE(max_err < 1e-6);

	// check throws error when stencil size is an odd number
	REQUIRE_THROWS(interpolation::generate_sinc_kernels<TestType>(step, 3));
}