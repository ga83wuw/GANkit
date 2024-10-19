# setup.py

from setuptools import setup, find_packages

setup(
    name='GANkit',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'matplotlib',
        'numpy',
        'opendatasets',
        'tqdm',
        'PyYAML',
    ],
    author='Georgios Athanasiou',
    author_email='georgiospathanasiou@yahoo.com',
    description='A personal toolkit for GANs.',
    keywords='GAN deep-learning pytorch',
    url='https://github.com/ga83wuw/GANkit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

