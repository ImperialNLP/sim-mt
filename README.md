# Multimodal simultaneous NMT

This repository is a stripped down clone of the upstream `nmtpytorch` repository.

## Contributors

- [Julia Ive][4] added all parts relating to reinforcement learning (RL) based
simultaneous MT and MMT.
- As part of her MSc. thesis, Veneta Haralampieva contributed layers & models for Transformers support
to simultaneous MT and MMT.

## Installation

The installation should be straightforward using anaconda. The below command will install the toolkit in `develop` mode into a newly created `simnmt` environment. This will allow your changes to the GIT checkout folder to be instantaneously reflected to the imported modules and executable scripts.

```
conda env create -f environment.yml
```

# Unsupervised reward in RL for MT

Code for the paper:

<b>[Exploring Supervised and Unsupervised Rewards in Machine Translation][1]</b>. Julia Ive, Zixu Wang, Marina Fomicheva, Lucia Specia (2021).
To appear in the Proceedings of EACL.


1. Follow the guidelines above to install the main code
 
2. Pre-train the actor (modify the paths in the config):

```bash
$ nmtpy train -C ./configs/unsupRL/en_de-cgru-nmt-bidir-base.conf
```

3. Train SAC with the unsupervised reward (modify the paths in the config, pretrained_file indicates the location of the pre-trained Actor):

```
$ nmtpy train -C ./configs/unsupRL/en_de-cgru-nmt-bidir-diyan.conf
```

The implementation of the Soft Actor-Critic framework follows the architecture and style of the [Deep-Reinforcement-Learning-Algorithms-with-PyTorch][2] library, developed by 
[Petros Christodoulou][3].

[1]: https://arxiv.org/abs/2102.11403
[2]: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
[3]: https://github.com/p-christ
[4]: https://julia-ive.github.io
 
