#include <iostream>
#include "kbdm.hpp"

int
main()
{
    auto data = VectorXcd::Constant(16, 0.6);
    auto dwell = 5e-4;
    auto m = data.size() / 2;
    auto l = m / 2;

    llckbdm::kbdm(
        data,
        dwell,
        m,
        l
    );

    return 0;
}