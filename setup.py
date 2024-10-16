from setuptools import setup, find_packages

setup(
    name="simulated_genomic_sequence_generator",
    version="0.1.0",
    author="RachidOunit",
    author_email="rouni001@cs.ucr.edu",  
    description="A package for generating simulated genomic sequences using a Transformer architecture.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rouni001/Simulated-Genomic-Sequence-Generator", 
    packages=find_packages(),
    install_requires=[
        "torch==2.4.1",
        "biopython==1.83",
        "numpy==1.24.4"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)

