from setuptools import setup, find_packages

setup(
    name='jaxkan',
    version='1.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==2.1.3',
        'scipy==1.14.1',
        'jax[cpu]==0.4.35',
        'optax==0.2.4',
        'flax==0.10.2',
    ],
    extras_require={
        'gpu': [
            'jax[cuda12]==0.4.35',
        ],
        'all': [
            'pytest==8.3.4',
            'jupyterlab==4.3.0',
            'matplotlib==3.9.2',
            'scikit-learn==1.5.2',
        ]
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
    python_requires='>=3.10',
)
