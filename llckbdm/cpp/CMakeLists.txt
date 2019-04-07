add_library(_kbdm MODULE kbdm.cpp kbdm.hpp)

add_definitions(-DEIGEN_USE_LAPACKE)

set(LINK_TARGETS Eigen3::Eigen  ${MKL_LIBRARIES})

add_executable(kbdm kbdm.cpp)

target_link_libraries(
        kbdm
        ${LINK_TARGETS}
)