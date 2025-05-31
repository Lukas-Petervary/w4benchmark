from setuptools import setup, find_packages

setup(
    name='w4benchmark',
    version='0.1.4',
    long_description="""\
An algorithm benchmarking tool built around the W4-11 dataset.

W4-11 Dataset © L. Goerigk and S. Grimme (2011).
Please cite:
This software uses the W4-11 dataset by Goerigk & Grimme (2011), which is intended for academic use only.
Please cite the original authors. Redistribution of the dataset may be subject to publisher restrictions.
""",
    long_description_content_type='text/plain',
    author='Lukas Petervary',
    url='https://github.com/Lukas-Petervary/w4benchmark/',
    license='CC BY-NC 4.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=2.2.4'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
    ],
)