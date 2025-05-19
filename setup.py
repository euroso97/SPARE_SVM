# svmtrainer_project/setup.py
from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read the requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='svmtrainer',
    version='0.1.1', # Increment version for updates
    author='<Your Name or Organization>',
    author_email='<your.email@example.com>',
    description='A Python package for training SVM models with hyperparameter tuning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='<URL to your package repository, e.g., GitHub>', # Optional
    packages=find_packages(exclude=['tests*', 'examples*']), # find_packages() automatically finds your 'svmtrainer' package
    install_requires=required,
    classifiers=[
        'Development Status :: 3 - Alpha', # Or '4 - Beta', '5 - Production/Stable'
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Or your chosen license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8', # Specify your minimum Python version
    keywords='svm machine-learning scikit-learn cross-validation hyperparameter-tuning',
    project_urls={ # Optional
        'Bug Reports': '<URL to your issue tracker>',
        'Source': '<URL to your source code>',
    },
)