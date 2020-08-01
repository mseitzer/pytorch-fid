import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-fid", # Replace with your own username
    version="1.0.0",
    author="Maximilian Seitzer",
    author_email="maximilian.seitzer@tuebingen.mpg.de",
    description="Package for calculating Frechet Inception Distance using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mseitzer/pytorch-fid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "scipy"
    ]
)
