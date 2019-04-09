#ifndef __HPP_CONSTANTS__
#define __HPP_CONSTANTS__

#define _USE_MATH_DEFINES
#include <cmath>

namespace llckbdm {
namespace constants {
    inline constexpr auto π = M_PI;
    inline constexpr auto I = std::complex<double>(0., 1.);
}  // namespace llckbdm
}  // namespace constants

#endif //__HPP_CONSTANTS__
