#include "kbdm.hpp"
#include <iostream>


int main() {
    std::cout << hello("World") << std::endl;
    return 0;
}

std::string hello(std::string name) {
    return "Hello, " + name;
}