add_subdirectory(interpolation_types)
add_subdirectory(interpolation_kernels)
add_subdirectory(interpolation)

add_executable(main main.cpp)
target_link_libraries(main interpolation_kernels interpolation)