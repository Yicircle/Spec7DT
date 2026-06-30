from setuptools import setup, find_packages

setup(
    name='Spec7DT',
    version='0.12.0',
    description='Spectral image handling package for 7-Dimensional Telescope users by Won-Hyeong Lee',
    author='Won-Hyeong Lee',
    author_email='wohy1220@gmail.com',
    url='https://github.com/Yicircle/Spec7DT',
    install_requires=[
        'numpy>=1.26,<2.7',
        'scipy>=1.12,<1.18',
        'astropy>=7.0,<8',
        'matplotlib>=3.8,<3.12',
        'photutils>=2.3,<3',
        'reproject>=0.20,<0.21',
        'pandas>=2.3,<3',
        'astroquery>=0.4.11,<0.5',
        'dustmaps>=1.0.14,<2',
    ],
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={
        "Spec7DT": [
            "reference/filter_curves/*.dat",
            "reference/psfs/*.fits",
            "reference/configs/*.swarp",
        ],
    },
    package_dir={'': 'src'},
    keywords=[''],
    python_requires='>=3.12,<3.13',
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.12'
    ],
)
