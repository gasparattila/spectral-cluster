## Spectral graph clustering implementation

Implementation of the graph clustering algorithm described in the article [Spectral redemption in clustering sparse networks](https://www.pnas.org/doi/10.1073/pnas.1312486110).

This library was developed as part of the [DYNASNET](https://dynasnet.renyi.hu/) project.

## Building

To download and install the library, run the following commands:
```
git clone https://github.com/gasparattila/spectral-cluster --recurse-submodules --shallow-submodules --depth 1
pip install ./spectral-cluster
```
Alternatively, you may download a pre-build binary for Python 3.11 on Windows:
```
pip install https://github.com/gasparattila/spectral-cluster/releases/download/v1.0/spectral_cluster-1.0-cp311-cp311-win_amd64.whl
```

## Usage

The Jupyter notebook [`example.ipynb`](example.ipynb) shows an example usage. See also the API documentation in [`spectral_cluster.py`](spectral_cluster.py).
