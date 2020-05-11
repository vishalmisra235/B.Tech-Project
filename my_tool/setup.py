import setuptools
import os


from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
	long_description = f.read()

#libfolder = os.path.dirname(os.path.realpath(__file__))
#requirementsPath = libfolder + '/requirements.txt'
#install_requires=[]
#if os.path.isfile(requirementsPath):
#	with open(requirementsPath) as f:
#		install_requires = f.read().splitlines()

setuptools.setup(
    name="metamorphic_tool",
    version="1.0.0",
    author="Vishal Misra",
    author_email="vishalmisra235@gmail.com",
    description="This is a prototype tool for metamorphic testing of image classifiers",
    long_description = long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/vishalmisra235/B.Tech-Project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
