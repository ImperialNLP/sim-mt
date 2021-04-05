# Multimodal simultaneous NMT

This repository is a stripped down clone of the upstream `nmtpytorch` repository.

## Contributors

- [Julia Ive][1] added all parts relating to reinforcement learning (RL) based
simultaneous MT and MMT.
- As part of her MSc. thesis, Veneta Haralampieva contributed layers & models for Transformers support
to simultaneous MT and MMT.

## Installation

The installation should be straightforward using anaconda. The below command will install the toolkit in `develop` mode into a newly created `simnmt` environment. This will allow your changes to the GIT checkout folder to be instantaneously reflected to the imported modules and executable scripts.

```
conda env create -f environment.yml
```

[1]: https://julia-ive.github.io
