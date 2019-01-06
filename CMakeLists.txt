cmake_minimum_required(VERSION 3.4.0)

project(kbdm)

set (CMAKE_CXX_STANDARD 11)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.9)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

set(PYBIND11_PYTHON_VERSION 3.6 CACHE STRING "")

if(DEFINED ENV{CONDA_PREFIX})
    message("Conda enviroment detected: $ENV{CONDA_PREFIX}")
    find_package(pybind11 REQUIRED)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
else()
    message(WARNING "Conda enviroment NOT detected. Third party submodules will be used.")
    add_subdirectory(third/pybind11)
    message("eigen: " ${CMAKE_CURRENT_SOURCE_DIR}/third/eigen)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/eigen")
endif()

add_subdirectory(llckbdm/cpp)
add_subdirectory(llckbdm/bindings)
