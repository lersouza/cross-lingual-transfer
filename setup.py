from setuptools import setup, find_packages
from crosslangt import VERSION

setup(
    name="crosslangt",
    packages=find_packages(),
    version=VERSION,
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.5',
        'pytorch-lightning==0.9.0',
        'transformers==4.30.0',
        'requests',
        'nltk==3.4.5',
        'tqdm==4.41.1',
        'lxml>=4.5.2',
        'deeppavlov==0.12.1',
        'requests==2.22.0'
    ]
)
