from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def main():
    setup(
        name='llckbdm',  
        version='0.2.4',
        description='Line List Clustering Krylov Basis Diagonalization Method implementation in Python',
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
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        keywords='kbdm mrs fitting hip llckbdm nmr',
        packages=find_packages(exclude=['contrib', 'docs']),
        license='MIT',
        python_requires='>=3.6',

        install_requires=['numpy', 'scipy', 'pandas', 'sklearn', 'attrs', 'hdbscan'],

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
