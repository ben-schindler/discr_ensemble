from os import path
from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="discr_ensemble",
    version=0.1,
    description="Multi-adversarial Learning in PyTorch.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ben-schindler/discr_ensemble",
    download_url = 'https://github.com/bayesiains/nflows/archive/v0.14.tar.gz',
    author="Benjamin Schindler",
    packages=find_packages(),
    license="MIT",
    install_requires=[
        "torch",
        "easydict",
        "torchinfo",
        "functorch"
    ],
    dependency_links=[],
)
