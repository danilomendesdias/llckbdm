cmake_minimum_required(VERSION 3.4.0)

project(kbdm)

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
