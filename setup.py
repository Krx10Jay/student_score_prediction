from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path)->List[str]:
    '''
    this function returns a list of the libraries needed as specified in your requirement file
    '''
    with open(file_path) as file_path_obj:
        requirements = file_path_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if '-e .' in requirements:
            requirements.remove("-e .")



setup(
    name="student_score prediction ML Project",
    version="0.0.1",
    author="Christian james",
    author_email= "jamescchidiebere@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)