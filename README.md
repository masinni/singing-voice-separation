# Singing Voice Separation from monaural recordings

## Introduction

This is a singing voice separation tool developed using unsupervised machine learning methods. It can seperate the singing voice and the background music from the original song.

It is a Python implementation of my thesis [Singing Voice Separation from Monaural Recordings
using Archetypal Analysis](https://pergamos.lib.uoa.gr/uoa/dl/object/3242634/file.pdf).

## Dependencies

* Python==3.10.6
* Numpy==1.21.5
* librosa==0.9.2
* soundfile==0.11.0
* matplotlib==3.5.1
* scikit-learn==1.1.3
* hlsvdpro==2.0.0

## Dataset

### MIR-1K Dataset

Music Information Retrieval, 1000 song clips ([MIR-1K](https://sites.google.com/site/unvoicedsoundseparation/mir-1k)), dataset for singing voice separation.

## Usage

### Separation

#### Robust Principal Component Analysis

To run separation using Robust Principal Component Analysis(RPCA), in terminal:

```bash
$ python svs_using_rpca.py
```

#### Archetypal Analysis with Sparsity Constrints

To run separation using Archetypal Analysis with Sparsity Constrints, in terminal:

```bash
$ python svs_using_archetypal_analysis.py
```

### Evaluation

Evaluation is happening when we execute each method.

## Referencies

* P.-S. Huang, S. D. Chen, P. Smaragdis, M. Hasegawa-Johnson, "Singing-Voice Separation From
Monaural Recordings Using Robust Principal Component Analysis" 2012.
