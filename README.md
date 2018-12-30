[![Build Status](https://travis-ci.com/danilomendesdias/llckbdm.svg?token=k6Bj4q2Uy7XrcNrLebfL&branch=master)](https://travis-ci.com/danilomendesdias/llckbdm)
[![codecov](https://codecov.io/gh/danilomendesdias/llckbdm/branch/master/graph/badge.svg?token=eOpnwCvmIt)](https://codecov.io/gh/danilomendesdias/llckbdm)

## Line List Clustering Krylov Basis Diagonalization Method
Core methods of Line List Clustering Krylov Basis Diagonalization Method written in Python 3.6.

In the current version, only KBDM is available. LLC-KBDM will be available soon.



#### Instalation

You can install it via pip:

`pip install llckbdm`


#### Development version

If you prefer, you can work with the development version by following these steps:

1. Make sure you have git, pip, virtualenv and Python 3.6 installed.

2. Clone the lastest version from Github

    `git clone https://github.com/danilomendesdias/llckbdm`

3. Setup your environment with virtualenv

    `virtualenv -p python3.6 .env`
    
4. Activate your virtual environment

    `source .env/bin/activate`
    
5. Install dependencies:

    `pip install -r requirements.txt`
    
6. Run tests:

    `py.test`
