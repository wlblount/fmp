from setuptools import setup, find_packages

setup(
    name='fmp',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'certifi',
        'requests',
        'pandas',
        'numpy',
        'matplotlib',
        'tqdm',
        'scikit-learn',
        'python-dotenv'
    ],
    author='BillBlount',
    author_email='',
    description='A package for accessing Financial Modeling Prep API',
    url='https://github.com/wlblount/fmp-repo',
)


