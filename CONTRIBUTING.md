# Contributor Guide

Contributions to this repository are welcome and greatly appreciated! We would love
for this package to grow and be supported by a larger community.

## Design principles

`cryojax` is built on [equinox](https://docs.kidger.site/equinox/). In short, `equinox` provides an interface to writing parameterized functions in `jax`. The core object of these parameterized functions is called a [Module](https://docs.kidger.site/equinox/api/module/module/) (yes, this takes inspiration from pytorch). `equinox` ships with features to interact with these `Module`s, and more generally with [pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) in `jax`. One of the most useful of these features, not found in `jax` itself, is a means of performing out-of-place updates on pytrees through `equinox.tree_at`.

Equinox also provides a recommended pattern for writing `Module`s: https://docs.kidger.site/equinox/pattern/. We think this is a good template for code readability, so `cryojax` tries to adhere to these principles as much as possible.

## What contributions fit into `cryojax`?

`cryojax` does not try to be a one-stop shop for cryo-EM analysis. The current scope of the package is outlined in the README.md. However, we would like to know what you would find helpful for your research, so if you have a contribution in mind please feel free to get in touch on the [Issue
Tracker](https://github.com/mjo22/cryojax/issues) and ask.

## Getting started

First, fork the library on GitHub. Then clone and install the library in development mode:

```
git clone https://github.com/your-username-here/cryojax.git
cd cryojax
python -m pip install -e .
```

Next, install the pre-commit hooks:

```
python -m pip install pre-commit
pre-commit install
```

This uses `ruff` to format and lint the code.

## Running tests

After making changes, make sure that the tests pass. In the `cryojax` base directory, run

```
python -m pip install -r tests/requirements.txt
python -m pytest
```

**If you are using a non-linux OS, the [`pycistem`](https://github.com/jojoelfe/pycistem) testing dependency cannot be installed**. In this case, in order to run the tests against [`cisTEM`](https://github.com/timothygrant80/cisTEM), run the testing [workflow](https://github.com/mjo22/cryojax/actions/workflows/testing.yml). This can be done manually or will happen automatically when a PR is opened.

## Building documentation

Again in the `cryojax` base directory, prepare to build the documentation by installing dependencies and pulling large-ish files from [git LFS](https://git-lfs.com/).

```
python -m pip install -r docs/requirements.txt
sudo apt-get install git-lfs  # If using macOS, `brew install git-lfs`
git lfs install; git lfs pull
```

Now, build the documentation with

```
mkdocs serve
```

and navigate to the local webpage by following the instructions in your terminal.

## How to submit changes

Now, if the tests and documentation look okay, push your changes and open a [Pull Request](https://github.com/mjo22/cryojax/pulls)!

## How to report a bug

Report bugs on the [Issue Tracker](https://github.com/mjo22/cryojax/issues).

When filing an issue, here are some guidelines that may be helpful to know:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or steps to
reproduce the issue. In particular, consider including a [Minimal, Reproducible
Example](https://stackoverflow.com/help/minimal-reproducible-example).

## How to request a feature

Feel free to request features on the [Issue
Tracker](https://github.com/mjo22/cryojax/issues).
