from setuptools import setup, find_packages

setup(
    name="classic_boids",
    version="0.1.0",
    python_requires=">=3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pytest",
    ],
)
