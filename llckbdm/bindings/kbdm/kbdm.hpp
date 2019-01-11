#ifndef __HELLO_H__
#define __HELLO_H__

#include <string>
#include <Eigen/Dense>

namespace llckbdm {

std::string kbdm(std::string name) {
  return "Hello, " + name;
}

} // namespace llckbdm

#endif