from skbuild import setup
from setuptools import find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def main():
    setup(
        name='llckbdm',  
        version='v0.1.2',
        description='Krylov Basis Diagonalization Method implementation in Python', 
        long_description=long_description,  
        long_description_content_type='text/markdown',
        url='https://github.com/danilomendesdias/llckbdm',  
        author='Danilo Mendes Dias Delfino da Silva',  
        author_email='danilomendesdias@gmail.com',  
        classifiers=[  
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'Topic :: Scientific/Engineering :: Physics',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
        ],
        keywords='kbdm mrs fitting hip',  
        packages=find_packages(exclude=['contrib', 'docs']),
        license='MIT',
        python_requires='>=3.6',

        install_requires=['cython', 'numpy', 'scipy', 'pandas', 'sklearn', 'hdbscan'],

        setup_requires=["pytest-runner"],
        tests_require=["pytest", "pytest-cov", "codecov"],

        data_files=[
            ('data', ['data/params_brain_sim_1_5T.csv'])
        ], 

        project_urls={ 
            'Source': 'https://github.com/danilomendesdias/llckbdm',
        },
    )


if __name__ == '__main__':
    main()
