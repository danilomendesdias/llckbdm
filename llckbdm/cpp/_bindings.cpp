#include "kbdm.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>


using namespace llckbdm;

PYBIND11_MODULE(_bindings, m) {
    m.def("kbdm", &kbdm, "Krylov Basis Diagonalization Method");
    m.def("_compute_U_matrices", &_compute_U_matrices, "");
    m.def("_solve_gep_svd", &_solve_gep_svd, "");
    m.def("_solve_svd", &_solve_svd, "");
    m.def("_solve_eig", &_solve_eig, "");
    m.def("_compute_U", &_compute_U, "");
    m.def("_compute_B", &_compute_B, "");
    m.def("_normalize_eigenvectors", &_normalize_eigenvectors, "");
    m.def("_compute_reduced_matrices", &_compute_reduced_matrices, "");
}
