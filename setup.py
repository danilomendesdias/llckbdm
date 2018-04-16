from setuptools import setup, find_packages

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='llckbdm',  # Required
    version='0.0.1',  # Required
    description='Krylov Basis Diagonalization Method implementation in Python',  # Requiredz
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/danilomendesdias/llckbdm',  # Optional
    author='Danilo Mendes Dias Delfino da Silva',  # Optional
    author_email='danilomendesdias@gmail.com',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='kbdm mrs fitting hip',  # Optional
    packages=find_packages(exclude=['contrib', 'docs']),  # Required

    install_requires=['numpy', 'scipy', 'pandas'],
    test_requires=['pytest'],

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #
    # If using Python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    package_data={  # Optional
        'llckbdm': ['package_data.dat'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[('my_data', ['data/data_file'])],  # Optional

    project_urls={  # Optional
        'Source': 'https://github.com/danilomendesdias/llckbdm',
    },
)

def main():

    setup(
            name='kbdm-lib',
            version='0.0.1',
            packages=find_packages(),
            url='https://github.com/danilomendesdias/kbdm-lib.git',
            license='MIT',
            author='danilomendesdias',
            author_email='danilomendesdias@gmail.com',
            description='Krylov Basis Diagonalization Method (KBDM) implementation written in Python',
            classifiers=[
                # How mature is this project? Common values are
                'Development Status :: 2 - Pre-Alpha',

                # Indicate who your project is intended for
                'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering :: Medical Science Apps.',
                'Topic :: Scientific/Engineering :: Physics',

                # Pick your license as you wish (should match "license" above)
                'License :: OSI Approved :: MIT License',

                # Specify the Python versions you support here. In particular, ensure
                # that you indicate whether you support Python 2, Python 3 or both.
                'Programming Language :: Python :: 3.6',
            ],
            install_requires=['numpy', 'scipy', 'pandas'],
            test_requires=['pytest']
    )


if __name__ == '__main__':
    main()
