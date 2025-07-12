from setuptools import setup, find_packages

setup(
    name='Spec7DT',
    version='0.9.9',
    description='Spectral image handling package for 7-Dimensional Telescope users by Won-Hyeong Lee',
    author='Won-Hyeong Lee',
    author_email='wohy1220@gmail.com',
    url='https://github.com/Yicircle/Spec7DT',
    install_requires=[
        'numpy',
        'astropy',
        'matplotlib',
        'seaborn',
        'pathlib',
        'photutils',
        'reproject',
    ],
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={
        "Spec7DT": ["reference/*"],
    },
    package_dir={'': 'src'},
    keywords=[''],
    python_requires='>=3.10',
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
)