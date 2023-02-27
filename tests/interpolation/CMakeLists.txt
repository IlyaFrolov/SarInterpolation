add_executable(test_interpolation test_interpolation.cpp)
target_link_libraries(test_interpolation interpolation_types interpolation Catch2::Catch2)

add_test(NAME "test interpolation" COMMAND test_interpolation)