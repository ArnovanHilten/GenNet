import os
import pathlib
from setuptools import find_namespace_packages, setup


def get_verified_absolute_path(path):
    """Verify and return absolute path of argument.
    Args:
        path : Relative/absolute path
    Returns:
        Absolute path
    """
    installed_path = os.path.abspath(path)
    if not os.path.exists(installed_path):
        raise RuntimeError("No valid path for requested component exists")
    return installed_path


def get_installation_requirments(file_path):
    """Parse pip requirements file.
    Args:
        file_path : path to pip requirements file
    Returns:
        list of requirement strings
    """
    with open(file_path, 'r') as file:
        requirements_file_content = \
            [line.strip() for line in file if
             line.strip() and not line.lstrip().startswith('#')]
    return requirements_file_content



# Classifiers for PyPI
pyaw_classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Scientist/Researchers/Developpers",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "License :: Apache 2.0"
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9"
]

current_dir = os.path.dirname(os.path.realpath(__file__))
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

required_packages = \
    get_installation_requirments(
        get_verified_absolute_path(
            os.path.join(current_dir, 'requirements_GenNet.txt'))
    )


setup(name='GenNet',
      description='Framework for interpretable neural networks',
      author='Arno van Hilten',
      author_email="a.vanhilten@erasmusmc.nl",
      url="https://github.com/ArnovanHilten/GenNet",
      include_package_data=True,
      install_requires=required_packages,
      packages=find_namespace_packages(),
      python_requires='>=3.5',
      long_description=long_description,
      classifiers=pyaw_classifiers,
      entry_points={'console_scripts': ['GenNet = GenNet_utils.GenNet:main']},
      platforms=['any'])