from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["torch"] # 这里填依赖包信息

setup(
    name="transformer-implementation-pytorch",
    version="1.0",
    author="Tao Xiang",
    author_email="tao.xiang@tum.de",
    description="A package of PyTorch implementation of transformers",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/leoxiang66/transformer-implementation-pytorch",
    packages=find_packages(),
    # Single module也可以：
    # py_modules=['timedd']
    install_requires=requirements,
    classifiers=[
	"Programming Language :: Python :: 3.7",
	"License :: OSI Approved :: MIT License",
    ],
)