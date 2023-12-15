from pathlib import Path

from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="tryondiffusion",
    version="0.1.0",
    license='Creative Commons BY-NC 4.0',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kailashahirwar/tryondiffusion',
    keywords='Unofficial implementation of TryOnDiffusion: A Tale Of Two UNets',
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "pillow",
        "matplotlib",
        "tqdm",
        "torchvision",
        "einops",
        "scipy"
    ]
)
