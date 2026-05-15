from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name('README.md').read_text(encoding='utf-8')

setup(
    name='Topyfic',  # the name of your package
    packages=find_packages(),  # same as above
    version='0.4.17',  # version number
    license='MIT',  # license type
    description='Topyfic is a Python package designed to identify reproducible latent dirichlet allocation (LDA) '
                'using leiden clustering and harmony for single cell epigenomics data',
    long_description=README,
    long_description_content_type='text/markdown',
    # short description
    author='Narges Rezaie',  # your name
    author_email='nargesrezaie80@gmail.com',  # your email
    url='https://github.com/mortazavilab/Topyfic',  # url to your git repo
    download_url='https://github.com/mortazavilab/Topyfic/archive/refs/tags/v0.4.17.tar.gz',  # link to the tar.gz file associated with this release
    keywords=['Cellular Programs', 'Latent Dirichlet allocation', 'single-cell multiome', 'single-cell RNA-seq',
              'gene regulatory network', 'Topic Modeling', 'single-nucleus RNA-seq'],  #
    python_requires='>=3.12',
    install_requires=[  # these can also include >, <, == to enforce version compatibility
        'adjustText>=1.3.0',
        'anndata>=0.12.0',
        'click>=8.3.0',
        'gseapy>=1.2.0',
        'h5py>=3.12.0',
        'joblib>=1.5.0',
        'leidenalg>=0.10.0',
        'matplotlib>=3.10.0',
        'networkx>=3.6',
        'numpy>=1.26.0',
        'obonet>=1.1.1',
        'pandas>=2.2.0',  # make sure the packages you put here are those NOT included in the base python distribution
        'plotly>=6.0.0',
        'PyYAML>=6.0.2',
        'reactome2py>=3.0.0',
        'scanpy>=1.12.0',
        'scikit-learn>=1.6.0',
        'scipy>=1.13.0',
        'seaborn>=0.13.0',
        'statsmodels>=0.14.6',
        'umap-learn>=0.5.12',
    ],
    extras_require={
        'dev': [
            'pytest>=8.3',
            'pytest-cov>=6.0',
            'pytest-xdist>=3.6',
            'sphinx>=8.0',
            'sphinx-bootstrap-theme>=0.8',
        ],
        'docs': [
            'sphinx>=8.0',
            'sphinx-bootstrap-theme>=0.8',
        ],
    },
    classifiers=[  # choose from here: https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
    ],
)
