from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """Read the requirements from a file and return them as a list."""

    requirement_list: List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            requirements = file.readlines()
            for req in requirements:
                requirement = req.strip()
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print(f"'requirements.txt' not found. No dependencies will be installed.")

    return requirement_list

setup(
    name='NetworkSecuritySystem',
    version='0.0.1',
    author='Sourav Halder',
    author_email='halder.sourav1996@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)