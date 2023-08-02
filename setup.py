from setuptools import setup

setup(
    name='Topyfic',  # the name of your package
    packages=['Topyfic'],  # same as above
    version='v0.4.3',  # version number
    license='MIT',  # license type
    description='Topyfic is a Python package designed to identify reproducible latent dirichlet allocation (LDA) '
                'using leiden clustering and harmony for single cell epigenomics data',
    # short description
    author='Narges Rezaie',  # your name
    author_email='nargesrezaie80@gmail.com',  # your email
    url='https://github.com/mortazavilab/Topyfic',  # url to your git repo
    download_url='https://github.com/mortazavilab/Topyfic/archive/refs/tags/v0.4.3.tar.gz',  # link to the tar.gz file associated with this release
    keywords=['Cellular Programs', 'Latent Dirichlet allocation', 'single-cell multiome', 'single-cell RNA-seq',
              'gene regulatory network', 'Topic Modeling', 'single-nucleus RNA-seq'],  #
    python_requires='>=3.8',
    install_requires=[  # these can also include >, <, == to enforce version compatibility
        'pandas>=1.4.4',  # make sure the packages you put here are those NOT included in the base python distribution
        'scikit-learn>=0.24.2',
        'pytest',
        'leidenalg',
        'setuptools',
        'anndata>=0.8.0',
        'click>=8.0.4',
        'gseapy>=1.0.1',
        'joblib>=1.2.0',
        'matplotlib>=3.5.2',
        'networkx>=2.8.4',
        'numpy>=1.23.2',
        'reactome2py>=3.0.0',
        'scanpy>=1.9.3',
        'scikit_learn>=1.2.2',
        'scipy>=1.9.1',
        'seaborn>=0.11.2',
        'statsmodels>=0.13.5',
        'adjustText'
        #'harmonypy'

    ],
    classifiers=[  # choose from here: https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research ',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
