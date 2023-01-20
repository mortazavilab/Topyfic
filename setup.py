from setuptools import setup

setup(
    name='Topyfic',  # the name of your package
    packages=['Topyfic'],  # same as above
    version='v0.1.3',  # version number
    license='MIT',  # license type
    description='Topyfic is a Python package designed to identify reproducible latent dirichlet allocation (LDA) '
                'using leiden clustering and harmony for single cell epigenomics data',
    # short description
    author='Narges Rezaie',  # your name
    author_email='nargesrezaie80@gmail.com',  # your email
    url='https://github.com/mortazavilab/Topyfic',  # url to your git repo
    download_url='https://github.com/mortazavilab/Topyfic/archive/refs/tags/V0.1.0.tar.gz',  # link to the tar.gz file associated with this release
    keywords=['Cellular Programs', 'Latent Dirichlet allocation', 'single-cell multiome', 'single-cell RNA-seq',
              'gene regulatory network'],  #
    python_requires='>=3.8',
    install_requires=[  # these can also include >, <, == to enforce version compatibility
        'pandas',  # make sure the packages you put here are those NOT included in the
        'numpy',  # base python distribution
        'scipy',
        'scikit-learn>=0.24.2',
        'matplotlib',
        'seaborn',
        'joblib',
        'pytest',
        'scanpy',
        'anndata',
        'leidenalg',
        'networkx',
        'scanpy',
        'setuptools',
        'statsmodels',

    ],
    classifiers=[  # choose from here: https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research ',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
