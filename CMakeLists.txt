cmake_minimum_required(VERSION 3.4.0)

project(kbdm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(third/pybind11)
add_subdirectory(third/eigen)

add_subdirectory(llckbdm/cpp)
add_subdirectory(llckbdm/bindings)
