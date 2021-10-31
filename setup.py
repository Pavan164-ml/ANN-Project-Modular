## setup.py is a python file, the presence of which is an indication that the module/package you are about 
## to install has likely been packaged and distributed with Distutils, which is the standard for distributing Python Modules.
## This allows you to easily install Python packages

from setuptools import setup

##“Unicode Transformation Format”, and the '8' means that 8-bit values are used in the encoding.
with open("README.md", "r", encoding="utf-8") as f :
    long_description = f.read()


setup(
    name = "src",
    version = "0.0.1",
    author = "Pavan164-ml",
    description = "This project is for my understanding on implementation of ANN with Tensorflow library and also have a taste of how the flow of modular coding works, which is used in the real world projects. There are many files which are interlinked with each other for their respective operations. If your looking for similar exposure for the project this repo might help you too!",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Pavan164-ml/ANN-Project-Modular",
    author_email = "pavanram2000@gmail.com",
    packages = ["src"],
    python_requires =">3.7",
    install_requires = [
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas",
        "sklearn"
    ]
    )     
