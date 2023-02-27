cmake_minimum_required(VERSION 3.22)

project(SarInterpolation LANGUAGES CXX)

set(BINARY_DIR "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${BINARY_DIR}/lib")
#set(CMAKE_CUDA_FLAGS "-arch=sm_30")
set(CMAKE_CUDA_FLAGS "-arch=sm_30")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
include(CTest)
include(GNUInstallDirs)

find_package(OpenMP REQUIRED)

if(NOT ${OpenMP_FOUND})
    message(SEND_ERROR "OpenMP library not found")
    return()
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(MPI REQUIRED)
if(NOT ${MPI_FOUND})
    message(SEND_ERROR "MPI library not found")
    return()
endif()

find_package(Catch2 REQUIRED)
if(NOT ${Catch2_FOUND})
    message(SEND_ERROR "Catch2 library not found")
    return()
endif()


add_definitions(-Dopenmp)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
add_subdirectory(third_party)
add_subdirectory(src)
add_subdirectory(tests)