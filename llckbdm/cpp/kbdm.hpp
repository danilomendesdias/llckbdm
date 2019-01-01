#include <iostream>

using namespace std;

string hello(string);

int main() {
    cout << hello("World") << endl;
    return 0;
}


string hello(string name) {
    return "Hello, " + name;
}