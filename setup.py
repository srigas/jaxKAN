from setuptools import setup, find_packages

setup(
    name='jaxkan',
    version='0.1.4',
    packages=find_packages(),
    install_requires=[
        'scipy==1.13.1',
        'numpy==1.26.4',
        'flax==0.8.3',
        'jax[cpu]==0.4.28',
        'optax==0.2.2',
    ],
    extras_require={
        'gpu': ['jax[cuda12]'],
    },
    author='Spyros Rigas, Michalis Papachristou',
    author_email='rigassp@gmail.com',
    description='A JAX-based implementation of Kolmogorov-Arnold Networks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/srigas/jaxkan',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
