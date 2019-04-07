#include "kbdm.hpp"
#include "mkl.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>



int main() {
    auto X = Eigen::Matrix2d::Zero(2, 2).eval();
    X(0, 0) = 1;
    X(1, 1) = 1;
    std::cout << X << std::endl;

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(X, Eigen::ComputeFullV);

    std::cout << svd.computeV();

    return 0;

}

std::string hello(std::string name) {
    return "Hello, " + name;
}