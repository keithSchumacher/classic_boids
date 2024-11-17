from setuptools import setup, find_packages

packages = find_packages(where="src")
print("Found packages:", packages)  # This will display the packages found during setup

setup(
    name="classic_boids",
    version="0.1.0",
    packages=packages,
    package_dir={"": "src"},
)
