# Multimodal simultaneous NMT

This repository is a stripped down clone of the upstream `nmtpytorch` repository.

## Installation

The installation should be straightforward using anaconda. The below command will install the toolkit in `develop` mode into a newly created `simnmt` environment. This will allow your changes to the GIT checkout folder to be instantaneously reflected to the imported modules and executable scripts.

```
conda env create -f environment.yml
```

## Dataset
- We assume Multi30K dataset for this work. The BPE-processed textual files are already under `./data`. The configuration files refer to these files for data loading. We validate on `val` set and report scores on `test_2016_flickr` for now.
- EN-DE direction is more challenging than EN-FR on this dataset, based on the BLEU/METEOR scores obtained by classical (consecutive) NMT systems. That's why, we can start by working on EN-DE direction.
- Visual features are convolutional feature maps stored in `.npy` files that you can [download from here](http://ozancaglayan.com/files/multi30k_features.tar.bz2). The shape of the stored tensor is `(N, 2048, 8, 8)` where `N` is the number of samples in a given train/test set.
    - For attentive models, that tensor is seen as `(N, 64, 2048)` where the 1st axis (64) is seen as spatial positions on top of which visual attention will be applied,
    - For simpler models, you may want to average pool the tensor to obtain `2048-dimensional` vectors per each image.

## Metrics

- Simultaneous MT metrics are implemented in `metrics/simnmt.py` file, the most widely acknowledged one being the average lagging i.e. `AVL`.
- A standalone [script](scripts/delay_analysis.py) is provided to compute metrics post-training, once you decode hypotheses with different
  simultaneous decoding approaches.

## Models

In `nmtpytorch` every model is defined in its own file and can be trained using the same `nmtpy` script without modification. An experiment is configured with a textual configuration file.

### SimultaneousNMT (snmt.py)

- This model implements classical NMT (i.e. consecutive NMT), which allows several simultaneous decoding approaches
  at test time. The model will also be used as the parent class for training-time models.
- **Unimodal example** -> [configs/en-de/word2word-min2/snmt-rnn-unimodal.conf](configs/en-de/word2word-min2/snmt-rnn-unimodal.conf)
- **Multimodal (attention) example** -> [configs/en-de/word2word-min2/snmt-rnn-multimodal.conf](configs/en-de/word2word-min2/snmt-rnn-multimodal.conf)

### SimultaneousWaitKNMT (snmt_waitk.py)
- This model implements the paper [6] i.e. training-time wait-k approach. The model simply redefines the `forward` pass
  and leaves everything the same as the parent `SimultaneousNMT` model.
- The only relevant argument in config is the `k` that should be passed to the `translator_args` dict.
- At test time, **same** `stranslate` pipeline may be used with different `k` values.
- **Unimodal example** -> [configs/en-de/word2word-min2/wait1-rnn-unimodal.conf](configs/en-de/word2word-min2/wait1-rnn-unimodal.conf)
- **Multimodal (attention) example** -> [configs/en-de/word2word-min2/wait1-rnn-multimodal.conf](configs/en-de/word2word-min2/wait1-rnn-multimodal.conf)

## Training, decoding and evaluation

**NOTE:** Before starting training, make sure that you set the `save_path` variable in the config files so that the
model files and tensorboard folders are correctly saved. Same goes for
all folders under the `[data]` section of configs.

### Random seeds
Random seed in config files are fixed to an arbitrary integer for now. When publishing results, you should set them to `0` and train 2-3
runs for each model to report mean and standard deviation of metrics. Alternatively, you can pass `train.seed:0` as the last argument to
`nmtpy train` command (as in the below example), to override the pre-determined seed from the command-line.

### Launch training
For training on GPU, you have to explicitly set the device ID with the
`CUDA_VISIBLE_DEVICES` environment variable. For running on CPU (which will take a lot more
time to finish), you can pass `-d cpu` to both `nmtpy train` and `nmtpy stranslate` commands.

```
# Example training command
CUDA_VISIBLE_DEVICES=0 nmtpy train -C configs/en-de/word2word-min2/snmt-rnn-unimodal.conf train.seed:0
```

### Post-training decoding

#### Vanilla (consecutive MT) greedy-search

Mostly used for baseline BLEU computations, for comparison purposes.

```
nmtpy stranslate -m 100 -s test_2016_flickr -f gs \
  -o <output file prefix> <best.bleu checkpoint>
```

#### wait_if_diff / wait_if_worse decoding (Cho and Esipova, 2016) [1]

The below command will decode with all combinations of n-init-tokens & delta & criteria
and output different files for each.

```
nmtpy stranslate -m 100 -s test_2016_flickr -f sgs \
  --n-init-tokens "1,2,3,4" --delta "1,2,3" --criteria "wait_if_diff,wait_if_worse" \
  -o <output file prefix> <best.bleu checkpoint>
```

#### Test time wait-k decoding (Ma et al., 2018) [6]
The below command will decode with k being 1,2,3,5 and 7, thus will output 5 sets of
output files.

```
nmtpy stranslate -m 100 -s test_2016_flickr -f wk \
  --n-init-tokens "1,2,3,5,7" -o <output file prefix> <best.bleu checkpoint>
```

#### Notes & Evaluation
When you decode a particular test set with one of the decoding approaches available,
you will see `*.gs, *.gs.raw, *.acts` files created as the output.

- `.gs` files are post-processed translations i.e. these should be used for BLEU computations.
- `.gs.raw` files do not apply post-processing such as BPE merging or de-hyphenization. This
  can be used for visualization or delay computation.
- `.acts` files contain the sequences of READ/WRITE actions committed during decoding.
  These files will be further used for delay metric computations.

If you follow the foldering/naming conventions provided in the config files,
you can use the `scripts/decode_*.sh` scripts, to end up with systematically named
output files, with same decoding parameters. These scripts should be run from the
folder which contains experiment subfolders such as `snmt-rnn-unimodal`.
Let us illustrate this part further:

```bash
$ cd ~/experiments/simnmt/nmtpy/w2w_models/en-de
$ ls
total 48K
drwxrwxr-x 2 ocaglaya ocaglaya  20K Apr 20 13:20 snmt-rnn-unimodal/
drwxrwxr-x 7 ocaglaya ocaglaya 4.0K Apr 20 14:13 tb_dir/
drwxrwxr-x 2 ocaglaya ocaglaya 4.0K Apr 20 14:57 wait1-rnn-multimodal/
drwxrwxr-x 2 ocaglaya ocaglaya 4.0K Apr 20 15:19 wait1-rnn-unimodal/
drwxrwxr-x 2 ocaglaya ocaglaya 4.0K Apr 20 15:20 wait2-rnn-multimodal/
drwxrwxr-x 2 ocaglaya ocaglaya 4.0K Apr 20 15:18 wait2-rnn-unimodal/

$ ~/experiments/simnmt/nmtpy/code/scripts/decode_test_wait_k.sh
  --> will run decoding on every possible best.bleu checkpoint file>
```

Once you finish decoding files, run the following script from the same folder to
get the results table:

```
$ <path to nmtpytorch code folder>/scripts/delay_analysis.py \
  -r <path to test_2016_flickr tokenized reference file>
```


## Papers

1. Can neural machine translation do simultaneous translation? (2016) https://arxiv.org/abs/1606.02012
2. Simultaneous Machine Translation using Deep Reinforcement (2016) https://pdfs.semanticscholar.org/ee1e/acd383ffaf0b4b00d7326dd4e6efc80dbb74.pdf
3. Learning to Translate in Real-time with Neural Machine Translation (EACL17) https://www.aclweb.org/anthology/E17-1099.pdf
4. Incremental Decoding and Training Methods for Simultaneous Translation in Neural Machine Translation (NAACL18) https://www.aclweb.org/anthology/N18-2079/
5. Prediction Improves Simultaneous Neural Machine Translation (EMNLP18) https://www.aclweb.org/anthology/D18-1337.pdf
6. STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework (ACL19) https://arxiv.org/abs/1810.08398
7. Monotonic Infinite Lookback Attention for Simultaneous Machine Translation (ACL19) https://arxiv.org/abs/1906.05218
8. Simultaneous Translation with Flexible Policy via Restricted Imitation Learning (ACL19) https://arxiv.org/abs/1906.01135
9. Speculative Beam Search for Simultaneous Translation (EMNLP19) https://arxiv.org/abs/1909.05421
10. Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation (EMNLP19) https://arxiv.org/abs/1909.01559
11. Simultaneous Neural Machine Translation using Connectionist Temporal Classification (2019) https://arxiv.org/pdf/1911.11933.pdf
12. Thinking Slow about Latency Evaluation for Simultaneous Machine Translation (2019) [METRIC] https://arxiv.org/pdf/1906.00048.pdf
13. Learning Coupled Policies for Simultaneous Machine Translation (2020) - http://arxiv.org/abs/2002.04306
14. Re-translation versus Streaming for Simultaneous Translation (2020) - https://arxiv.org/pdf/2004.03643v2.pdf
15. Efficient Wait-k Models for Simultaneous Machine Translation (2020) - https://arxiv.org/pdf/2005.08595.pdf
16. Monotonic multihead attention (2020) - https://arxiv.org/pdf/1909.12406.pdf
17. Simultaneous Translation Policies: From Fixed to Adaptive (2020) - https://arxiv.org/pdf/2004.13169.pdf