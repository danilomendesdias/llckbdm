cmake_minimum_required(VERSION 3.4.0)

project(kbdm)

set (CMAKE_CXX_STANDARD 17)

set (LLCKBDM_LIB_PATH $ENV{LLCKBDM_LIB_PATH})
if(NOT LLCKBDM_LIB_PATH)
    set(LLCKBDM_LIB_PATH llckbdm)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

if(APPLE)
    message("Apple detected")
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.10)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

set(PYBIND11_PYTHON_VERSION 3.6 CACHE STRING "")

if(DEFINED ENV{CONDA_PREFIX})
    message("Conda enviroment detected: $ENV{CONDA_PREFIX}")
    set(Eigen3_DIR "$ENV{CONDA_PREFIX}/share/eigen3/cmake")
    set(pybind11_DIR "$ENV{CONDA_PREFIX}/share/cmake/pybind11")
    set(INCLUDE_ROOT "$ENV{CONDA_PREFIX}/include")
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    find_package(pybind11 REQUIRED)
    find_package(MKL)
else()
    message(WARNING "Conda enviroment NOT detected. Third party submodules will be used.")
    set(Eigen3_DIR ${PROJECT_SOURCE_DIR}/third/eigen)
    include_directories(${Eigen3_DIR})
    add_subdirectory(third/pybind11)

endif()

if (MKL_FOUND)
    message("MKL_FOUND: ${MKL_LIBRARIES}")
endif()
message("-- Eigen3_DIR: " ${Eigen3_DIR})
message("-- pybind11_DIR: " ${pybind11_DIR})

add_subdirectory(llckbdm/cpp)
