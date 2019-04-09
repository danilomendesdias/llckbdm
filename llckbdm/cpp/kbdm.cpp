#include "kbdm.hpp"
#include "constants.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <tuple>
#include <exception>
#include <cmath>

using namespace Eigen;

namespace llckbdm
{

std::tuple<ArrayXd, ArrayXd, ArrayXd, ArrayXd>
kbdm(
    VectorConstRef data,
    double dwell,
    int m,
    int l,
    int p
)
{
    auto[U0, Up_1, Up] = _compute_U_matrices(data, m, p);

    auto[mu, B_norm] = _solve_gep_svd(U0, Up_1, Up, l);

    auto D_sqrt = (B_norm.transpose() * data.block(0, 0, m, 1)).eval();

    ArrayXcd D = D_sqrt.array().square();

    ArrayXd A = D.abs();
    ArrayXd PH = D.arg();

    ArrayXcd Omega = -constants::I * mu.array().log() / dwell;

    ArrayXd F = Omega.real() / (2 * constants::pi);

    ArrayXd T2 = 1. / Omega.imag();

    return std::make_tuple(A, T2, F, PH);
}

std::tuple<MatrixXcd, MatrixXcd, MatrixXcd>
_compute_U_matrices(VectorConstRef data, int m, int p)
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
_solve_svd(MatrixConstRef Up_1)
{
    auto bcd_svd_solver = Eigen::BDCSVD<Eigen::MatrixXcd>(Up_1, Eigen::ComputeFullU | Eigen::ComputeFullV);

    MatrixXcd L = bcd_svd_solver.matrixU();
    VectorXd s = bcd_svd_solver.singularValues();
    MatrixXcd R = bcd_svd_solver.matrixV();

    return std::make_tuple(L, s, R);
}

std::tuple<VectorXcd, MatrixXcd>
_solve_eig(MatrixConstRef U)
{
    auto eigen_solver = ComplexEigenSolver<MatrixXcd>(U);

    MatrixXcd P = eigen_solver.eigenvectors();
    VectorXcd mu = eigen_solver.eigenvalues();

    return std::make_tuple(mu, P);
}

MatrixXcd
_compute_U(MatrixConstRef Dsqi_, MatrixConstRef L_h_, MatrixConstRef Up, MatrixConstRef R_)
{

    return Dsqi_ * L_h_ * Up * R_ * Dsqi_;
}

MatrixXcd
_compute_B(MatrixConstRef R_, MatrixConstRef Dsqi_, MatrixConstRef P)
{
// Obtaining eigenvectors at krylov basis
    return R_ * Dsqi_ * P;
}

std::tuple<MatrixXcd, MatrixXcd, MatrixXcd>
_compute_reduced_matrices(MatrixConstRef L, VectorXd const& s, MatrixConstRef R, int m, int l)
{
    MatrixXcd Dsqi = s.cwiseSqrt().cwiseInverse().asDiagonal();

    MatrixXcd Dsqi_ = Dsqi.block(0, 0, l, l);
    MatrixXcd L_h_ = L.block(0, 0, m, l).adjoint();
    MatrixXcd R_ = R.block(0, 0, m, l);

    return std::make_tuple(Dsqi_, L_h_, R_);
}


std::tuple<VectorXcd, MatrixXcd>
_solve_gep_svd(MatrixConstRef U0, MatrixConstRef Up_1, MatrixConstRef Up, int l)
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
_normalize_eigenvectors(MatrixXcd B, MatrixConstRef U0)
{
    auto l = B.cols();

    for (auto k = 0; k < l; ++k)
    {
        auto norm_factor = std::sqrt((B.col(k).transpose() * U0 * B.col(k)).value());
        B.col(k) = B.col(k) / norm_factor;
    }

    return B;
}

} // namespace llckbdm
