from setuptools import setup, find_packages

setup(
    name='mlpynem',
    version='0.0.1',
    packages=find_packages('mlpynem'),
    install_requires=[
        'numpy==1.23.5',
        'scikit-learn',
        'jupyter',
        'matplotlib',
        'lmfit',
        'tqdm'],
    

    # metadata to display on PyPI
    authors='Adrien Teurtrie, Hugo Lourenco-Martins'
)
    