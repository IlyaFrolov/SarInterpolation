add_executable(test_interpolation_kernels test_interpolation_kernels.cpp)
target_link_libraries(test_interpolation_kernels interpolation_types interpolation_kernels Catch2::Catch2 Eigen3::Eigen)

add_test(NAME "test interpolation kernels" COMMAND test_interpolation_kernels)