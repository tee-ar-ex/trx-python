# tee-ar-ex
Official repository to present the TRX (tee-ar-ex) python implementation, discuss issues and keep track of everything.
Anyone is free to contribute (issues, code reviews, pull requests).
Due to possible changes in implementation and possibily specifications, we recommend using a virtual environement.

The technical specifications of the file format is available [here](https://github.com/tee-ar-ex/trx-spec/blob/master/specifications.md).

The library can be installed locally by using the following commands:
```
export TRX_MINIMAL_INSTALL=1
pip install -e .
```
If you want to use the scripts and the functions related to the Stateful
Tractogram using the following commands:
```
pip install packaging>=19.0
pip install cython>=0.29.24
pip install -e .
```
If you want to switch from a minimal (w/o Dipy) to a full (with Dipy) install
you may need to add this before the above commands.
```
export TRX_MINIMAL_INSTALL=0
```