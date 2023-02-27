set(CMAKE_CXX_STANDARD 14)

add_library(interpolation_kernels SHARED interpolation_kernels.hpp interpolation_kernels.cpp)
target_link_libraries(interpolation_kernels interpolation_types)
target_include_directories(interpolation_kernels INTERFACE .)
