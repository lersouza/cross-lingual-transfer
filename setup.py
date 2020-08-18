from setuptools import setup, find_packages
from crosslangt import VERSION

setup(
    name="crosslangt",
    packages=find_packages(),
    version=VERSION,
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.5',
        'pytorch-lightning==0.8.5',
        'transformers==3.0.2',
        'requests',
        'nltk',
        'tqdm',
        'lxml'
    ]
)
