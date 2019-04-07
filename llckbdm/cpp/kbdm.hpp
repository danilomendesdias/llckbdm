#ifndef __HPP_KBDM__
#define __HPP_KBDM__

#include <Eigen/Dense>
#include <tuple>

using namespace Eigen;

namespace llckbdm
{

constexpr auto I = std::complex<double>(0., 1.);

std::tuple<MatrixXcd, MatrixXcd, MatrixXcd>
_compute_U_matrices(VectorXcd const &data, int m, int p);

std::tuple<VectorXcd, MatrixXcd>
_solve_gep_svd(MatrixXcd const& U0, MatrixXcd const& Up_1, MatrixXcd const& Up, int l, double q);

MatrixXcd
_normalize_eigenvectors(MatrixXcd const &B, MatrixXcd const &U0);

std::tuple<ArrayXd, ArrayXd, ArrayXd, ArrayXd>
kbdm(
    VectorXcd const &data,
    double dwell,
    int m,
    int l,
    double q = 0.0,
    int p = 1
);

std::tuple<MatrixXcd, VectorXd, MatrixXcd>
_solve_svd(MatrixXcd const &Up_1);

std::tuple<VectorXcd, MatrixXcd>
_solve_eig(MatrixXcd const &U);

MatrixXcd
_compute_U(MatrixXcd const& Dsqi_, MatrixXcd const& L_h_, MatrixXcd const& Up, MatrixXcd const& R_);

MatrixXcd
_compute_B(MatrixXcd const& R_, MatrixXcd const& Dsqi_, MatrixXcd const& P);

std::tuple<MatrixXcd, MatrixXcd, MatrixXcd>
_compute_reduced_matrices(MatrixXcd const& L, VectorXd const& s, MatrixXcd const& R, int m, int l);

} // namespace llckbdm

#endif //__HPP_KBDM__