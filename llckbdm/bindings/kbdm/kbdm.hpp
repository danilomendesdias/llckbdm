#ifndef __HPP_KBDM__
#define __HPP_KBDM__

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <tuple>
#include <exception>
#define _USE_MATH_DEFINES
#include <cmath>

using namespace Eigen;

auto I = std::complex<double>(0., 1.);

namespace llckbdm
{

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
)
{
    auto[U0, Up_1, Up] = _compute_U_matrices(data, m, p);

    auto[mu, B_norm] = _solve_gep_svd(U0, Up_1, Up, l, q);

    auto D_sqrt = (B_norm.transpose() * data.block(0, 0, m, 1)).eval();

    ArrayXcd D = D_sqrt.array().square();

    ArrayXd A = D.abs();
    ArrayXd PH = D.arg();

    ArrayXcd Omega = -I * mu.array().log() / dwell;

    ArrayXd F = Omega.real() / (2 * M_PI);

    ArrayXd T2 = Omega.imag().inverse();

    return std::make_tuple(A, T2, F, PH);
}

std::tuple<MatrixXcd, MatrixXcd, MatrixXcd>
_compute_U_matrices(VectorXcd const &data, int m, int p)
{
    auto U0 = MatrixXcd(m, m);
    auto Up = MatrixXcd(m, m);
    auto Up_1 = MatrixXcd(m, m);

    for (auto i = 0; i < m; ++i)
    {
        for (auto j = 0; j < m; ++j)
        {
            U0(i, j) = data[i + j];
            Up(i, j) = data[i + j + p];
            Up_1(i, j) = data[i + j + p - 1];
        }
    }

    return std::make_tuple(U0, Up_1, Up);
}

std::tuple<MatrixXcd, VectorXd, MatrixXcd>
_solve_svd(MatrixXcd const &Up_1)
{
    auto bcd_svd_solver = BDCSVD<MatrixXcd>(Up_1, ComputeThinU | ComputeThinV);

    MatrixXcd L = bcd_svd_solver.matrixU();
    VectorXd s = bcd_svd_solver.singularValues();
    MatrixXcd R = bcd_svd_solver.matrixV();

    return std::make_tuple(L, s, R);
}

std::tuple<VectorXcd, MatrixXcd>
_solve_eig(MatrixXcd const &U)
{
    auto eigen_solver = ComplexEigenSolver<MatrixXcd>(U);

    MatrixXcd P = eigen_solver.eigenvectors();
    VectorXcd mu = eigen_solver.eigenvalues();

    return std::make_tuple(mu, P);
}

MatrixXcd
_compute_U(MatrixXcd const& Dsqi_, MatrixXcd const& L_h_, MatrixXcd const& Up, MatrixXcd const& R_)
{

    return Dsqi_ * L_h_ * Up * R_ * Dsqi_;
}

MatrixXcd
_compute_B(MatrixXcd const& R_, MatrixXcd const& Dsqi_, MatrixXcd const& P)
{
// Obtaining eigenvectors at krylov basis
    return R_ * Dsqi_ * P;
}

std::tuple<MatrixXcd, MatrixXcd, MatrixXcd>
_compute_reduced_matrices(MatrixXcd const& L, VectorXd const& s, MatrixXcd const& R, int m, int l)
{
    MatrixXcd Dsqi = s.cwiseSqrt().cwiseInverse().asDiagonal();

    MatrixXcd Dsqi_ = Dsqi.block(0, 0, l, l);
    MatrixXcd L_h_ = L.block(0, 0, m, l).adjoint();
    MatrixXcd R_ = R.block(0, 0, m, l);

    return std::make_tuple(Dsqi_, L_h_, R_);
}


std::tuple<VectorXcd, MatrixXcd>
_solve_gep_svd(MatrixXcd const& U0, MatrixXcd const& Up_1, MatrixXcd const& Up, int l, double q)
{
    auto m = U0.rows();

    auto[L, s, R] = _solve_svd(Up_1);

    auto [Dsqi_, L_h_, R_] = _compute_reduced_matrices(L, s, R, m, l);

    auto U = _compute_U(Dsqi_, L_h_, Up, R_);

    auto[mu, P] = _solve_eig(U);

    auto B = _compute_B(R_, Dsqi_, P);

    auto B_norm = _normalize_eigenvectors(B, U0);

    return std::make_tuple(mu, B_norm);
}

MatrixXcd
_normalize_eigenvectors(MatrixXcd const &B, MatrixXcd const &U0)
{
    auto l = B.cols();

    auto B_norm = B;

    for (auto k = 0; k < l; ++k)
    {
        B_norm.col(k) = B.col(k) * ((B.col(k).transpose() * U0 * B.col(k)).inverse().sqrt());
    }

    return B_norm;
}

} // namespace llckbdm

#endif //__HPP_KBDM__