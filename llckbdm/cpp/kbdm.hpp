#ifndef __HPP_KBDM__
#define __HPP_KBDM__

#include <Eigen/Dense>
#include <tuple>

using namespace Eigen;

namespace llckbdm
{

typedef const Ref<const MatrixXcd>& MatrixConstRef;
typedef const Ref<const VectorXcd>& VectorConstRef;

std::tuple<MatrixXcd, MatrixXcd, MatrixXcd>
_compute_U_matrices(VectorConstRef data, int m, int p);

std::tuple<VectorXcd, MatrixXcd>
_solve_gep_svd(MatrixConstRef U0, MatrixConstRef Up_1, MatrixConstRef Up, int l);

MatrixXcd
_normalize_eigenvectors(MatrixXcd B, MatrixConstRef U0);

std::tuple<ArrayXd, ArrayXd, ArrayXd, ArrayXd>
kbdm(
    VectorConstRef data,
    double dwell,
    int m,
    int l,
    int p = 1
);

std::tuple<MatrixXcd, VectorXd, MatrixXcd>
_solve_svd(MatrixConstRef Up_1);

std::tuple<VectorXcd, MatrixXcd>
_solve_eig(MatrixConstRef U);

MatrixXcd
_compute_U(MatrixConstRef Dsqi_, MatrixConstRef L_h_, MatrixConstRef Up, MatrixConstRef R_);

MatrixXcd
_compute_B(MatrixConstRef R_, MatrixConstRef Dsqi_, MatrixConstRef P);

std::tuple<MatrixXcd, MatrixXcd, MatrixXcd>
_compute_reduced_matrices(MatrixConstRef L, VectorXd const& s, MatrixConstRef R, int m, int l);

} // namespace llckbdm

#endif //__HPP_KBDM__