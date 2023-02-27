set(CMAKE_CXX_STANDARD 14)

add_library(interpolation SHARED interpolation.hpp interpolation.cpp)

target_link_libraries(interpolation interpolation_types interpolation_kernels)
target_include_directories(interpolation_types INTERFACE .)
