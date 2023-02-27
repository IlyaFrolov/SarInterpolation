#define EIGEN_DONT_PARALLELIZE 
#include <omp.h>
#include "interpolation.hpp"
#include <math.h>
#include <exception>

namespace interpolation
{
	using Eigen::placeholders::last;
	using Eigen::seq;

	template<typename T>
	void interpolate_cpu(const Matrix<T>& kernels, const Matrix<T>& samples, const Matrix<double>& new_indexes, Matrix<T>& result)
	{
		int N = samples.cols();

		result.resize(new_indexes.rows(), new_indexes.cols());

		Eigen::initParallel();
		int n_kernels = kernels.rows();
		int stencil_size = kernels.cols();

		int half_stencil_size = stencil_size / 2;

		int stencil_middle = stencil_size / 2 - 1;

		if (n_kernels < 2)
			throw std::logic_error("expected at least 2 kernels");

		double kernels_step = 1.0 / (static_cast<double>(n_kernels) - 1.0);
		//int n_procs = 1;
		//n_procs = omp_get_num_procs();
        #pragma omp parallel shared(kernels, samples, new_indexes, result) //num_threads=n_procs
		{
			int thread_id = 0;
#ifdef _OPENMP
			thread_id = omp_get_thread_num();
#endif
			for (int line_i = thread_id; line_i < new_indexes.rows(); line_i += omp_get_num_procs())
			{
				for (int i = 0; i < new_indexes.cols(); i += 1)
				{
					double new_idx = new_indexes(line_i, i);
					int sample_i = static_cast<int>(floor(new_idx));
					double shift = new_idx - static_cast<double>(sample_i);
					int kernel_i = static_cast<int>(round(shift / kernels_step));

					int sample1 = sample_i - stencil_size / 2 + 1, sample2 = sample_i + stencil_size / 2;
					int kernel_sample1 = 0, kernel_sample2 = stencil_size - 1;

					if ((sample_i + half_stencil_size) < 0)
					{
						result(line_i, i) = 0;
						continue;
					}
					if ((sample_i - half_stencil_size + 1) > (N - 1))
					{
						result(line_i, i) = 0;
						continue;
					}
					if ((sample_i - half_stencil_size + 1) < 0)
					{
						sample1 = 0;
						kernel_sample1 = -(sample_i - half_stencil_size + 1);
					}
					if ((sample_i + half_stencil_size) > (N - 1))
					{
						sample2 = N - 1;
						kernel_sample2 = stencil_size - 1 - ((sample_i + half_stencil_size) - (N - 1));
					}

					result(line_i, i) = samples(line_i, seq(sample1, sample2)).dot(kernels(kernel_i, seq(kernel_sample1, kernel_sample2)));
				}
			}
		}

	}
}

template void interpolation::interpolate_cpu<float>(const interpolation::Matrix<float>&, const interpolation::Matrix<float>&, const interpolation::Matrix<double>&, interpolation::Matrix<float>&);
template void interpolation::interpolate_cpu<double>(const interpolation::Matrix<double>&, const interpolation::Matrix<double>&, const interpolation::Matrix<double>&, interpolation::Matrix<double>&);
template void interpolation::interpolate_cpu<std::complex<float>>(const interpolation::Matrix<std::complex<float>>&, const interpolation::Matrix<std::complex<float>>&, const interpolation::Matrix<double>&, interpolation::Matrix<std::complex<float>>&);
template void interpolation::interpolate_cpu<std::complex<double>>(const interpolation::Matrix<std::complex<double>>&, const interpolation::Matrix<std::complex<double>>&, const interpolation::Matrix<double>&, interpolation::Matrix<std::complex<double>>&);
