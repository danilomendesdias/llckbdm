#include <pybind11/pybind11.h>
#include "kbdm/kbdm.hpp"
#include <Eigen/Dense>

using namespace llckbdm;

PYBIND11_MODULE(_bindings, m) {
    m.def("kbdm", &kbdm, "A function to say hello");
}
