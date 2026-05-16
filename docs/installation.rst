Installation
============

Topyfic currently targets Python 3.12 through 3.14.

By default, Topyfic now prefers the PyTorch backend when PyTorch is installed. With the default ``device=auto`` setting, Topyfic will use CUDA on NVIDIA systems, MPS on Apple Silicon when available, and PyTorch CPU otherwise. If PyTorch is not installed, Topyfic falls back to the sklearn backend.


Install from PyPi (recommended)
-------------------------------

Install the most recent release, run

``pip install Topyfic``

If you want the default accelerated backend, install PyTorch separately after installing Topyfic.

For Apple Silicon, the standard wheel is usually sufficient:

``pip install torch``

For Linux NVIDIA systems, install a wheel that matches the server's CUDA runtime and driver stack. For example:

``pip install torch --index-url https://download.pytorch.org/whl/cu128``

If PyTorch is not installed, Topyfic will continue to use the sklearn backend.


Install with the most recent commits
------------------------------------

git cloning the `Topyfic repository <https://github.com/mortazavilab/Topyfic>`_, going to the Topyfic directory, run

``pip install .``


Install for local development
-----------------------------

To work on the package locally, install it in editable mode:

``pip install -e .``

To install development and test dependencies as well:

``pip install -e .[dev]``


Backend selection
-----------------

The Python API and CLI now default to the torch backend when PyTorch is installed.

You can still select explicit backup paths from the command line:

- force torch on CPU: ``python -m Topyfic.main train_model --backend torch --device cpu ...``
- force sklearn: ``python -m Topyfic.main train_model --backend sklearn ...``

If you omit ``--backend``, Topyfic resolves the default automatically.

