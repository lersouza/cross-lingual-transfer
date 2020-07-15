from setuptools import setup, find_packages
from crosslangt import VERSION

setup(
    name="crosslangt",
    packages=find_packages(),
    version=VERSION,
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.5',
        'pytorch_lightning==0.8.5',
        'transformers==2.11.0',
        'requests',
        'nltk',
        'tqdm',
        'lxml'
    ]
)
