from setuptools import setup, find_packages

setup(
    name="diamond_pricing_pipeline",
    version="0.1",
    author="Alex",
    author_email="alex@inthedata.stream",
    description="A Python package for diamond price prediction with XGBoost and Linear Regression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hrokr/diamond_pricing_pipeline",  # Update with your repository URL
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

