from setuptools import setup

setup(
    name='fmp',
    version='0.1',
    py_modules=['fmp'],
    install_requires=[
        'certifi',
        'requests',
        'pandas',
        'numpy',
        'matplotlib',
        'tqdm',
        'scikit-learn',
        'python-dotenv',
        'ipython',
    ],
    author='BillBlount',
    author_email='',
    description='A package for accessing Financial Modeling Prep API',
    url='https://github.com/wlblount/fmp-repo',
)


