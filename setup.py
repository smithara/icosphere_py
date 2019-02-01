from setuptools import setup, find_packages

with open('icosphere/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

setup(
    name='icosphere_py',
    packages=find_packages(),
    version=version,
    description='Generating a spherical grid',
    url='https://github.com/smithara/icosphere_py',
    author='Ashley Smith',
    author_email='ashley.smith@ed.ac.uk',
    license='MIT',
    install_requires=['pandas']
)
