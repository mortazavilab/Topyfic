from setuptools import setup

setup(
    name='Topyfic',  # the name of your package
    packages=['Topyfic'],  # same as above
    version='v0.4.11',  # version number
    license='MIT',  # license type
    description='Topyfic is a Python package designed to identify reproducible latent dirichlet allocation (LDA) '
                'using leiden clustering and harmony for single cell epigenomics data',
    # short description
    author='Narges Rezaie',  # your name
    author_email='nargesrezaie80@gmail.com',  # your email
    url='https://github.com/mortazavilab/Topyfic',  # url to your git repo
    download_url='https://github.com/mortazavilab/Topyfic/archive/refs/tags/v0.4.11.tar.gz',  # link to the tar.gz file associated with this release
    keywords=['Cellular Programs', 'Latent Dirichlet allocation', 'single-cell multiome', 'single-cell RNA-seq',
              'gene regulatory network', 'Topic Modeling', 'single-nucleus RNA-seq'],  #
    python_requires='>=3.9',
    install_requires=[  # these can also include >, <, == to enforce version compatibility
        'pandas>=1.4.4',  # make sure the packages you put here are those NOT included in the base python distribution
        'scikit-learn>=0.24.2',
        'pytest',
        'leidenalg',
        'setuptools',
        'anndata>=0.9.2',
        'click>=8.0.4',
        'gseapy>=1.0.5',
        'joblib>=1.3.2',
        'matplotlib>=3.7.2',
        'networkx>=2.8.4',
        'numpy>=1.24.4',
        'reactome2py>=3.0.0',
        'scanpy>=1.9.5',
        'scikit_learn>=1.3.0',
        'scipy>=1.11.2',
        'seaborn>=0.11.2',
        'statsmodels>=0.14.0',
        'adjustText>=0.8',
        'PyYAML>=6.0.1',
        'click>=8.1.7',
        'umap',
        'obonet',
        'plotly',
        'h5py',

    ],
    classifiers=[  # choose from here: https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research ',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)
