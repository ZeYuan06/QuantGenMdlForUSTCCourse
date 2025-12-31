# QuantGenMdl

This repository contains the official Python implementation of [*Generative Quantum Machine Learning via Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2310.05866), an article by [Bingzhi Zhang](https://sites.google.com/view/bingzhi-zhang/home), [Peng Xu](https://francis-hsu.github.io/), [Xiaohui Chen](https://the-xiaohuichen.github.io/), and [Quntao Zhuang](https://sites.usc.edu/zhuang).

## Citation
```
@misc{zhang2023generative,
      title={Generative quantum machine learning via denoising diffusion probabilistic models}, 
      author={Bingzhi Zhang and Peng Xu and Xiaohui Chen and Quntao Zhuang},
      year={2023},
      eprint={2310.05866},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

## Prerequisite
The simulation of quantum circuit is performed via the [TensorCircuit](https://tensorcircuit.readthedocs.io/en/latest/#) package. We explored with the [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) backends during development. As a result, all three are required in order to run all the notebooks presented in this repository. Use of GPU is not required, but highly recommended.

Additionally, the packages [POT](https://pythonot.github.io/) and [OTT](https://ott-jax.readthedocs.io/en/latest/) are required for the computation of Wasserstein distance, [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/stable/) is used for speeding up certain evaluation, and [Optax](https://github.com/google-deepmind/optax) is needed for optimization with the JAX backend.

## File Structure
Notebooks in this repository can be used to reproduce the experiment presented in the paper. Their file names are self-explanatory:
| Notebook        | Generation Task   | Backend     |
| :---            | :----             | :---        |
| `QDDPM_circle`  | Circular States   | TensorFlow  |
| `QDDPM_cluster` | Clustered State   | PyTorch     |
| `QDDPM_noise`   | Correlated Noise  | TensorFlow  |
| `QDDPM_phase`   | Many-body Phase   | JAX         |

In addition to these, the two files `QDT_training.ipynb` and `QGAN_training.ipynb` show the training process of our benchmark models (Quantum Direct Transport and Quantum GAN, respectfully), and both utilize the JAX backend.

Lastly, code in `bloch_visualize.ipynb` were used to generate the Bloch sphere visualizations used in the paper.

## MNIST01 (0 vs 1) pipeline

This course workspace includes a minimal end-to-end MNIST01 (binary 0/1) pipeline:

- Prepare dataset (download MNIST, downsample to 14×14, PCA to 8 dims):
      - scripts/mnist01_prepare.py
- Encode latent vectors into $n=8$-qubit product states:
      - scripts/mnist01_encode_states.py
- Train a small CNN classifier on real images (used for evaluation features):
      - scripts/mnist01_train_classifier.py
- Train / generate with models:
      - QGAN: scripts/mnist01_train_qgan.py, scripts/mnist01_generate_qgan.py
      - QDT: scripts/mnist01_train_qdt.py, scripts/mnist01_generate_qdt.py
      - QDDPM: scripts/mnist01_make_diffusion_qddpm.py, scripts/mnist01_train_qddpm.py, scripts/mnist01_generate_qddpm.py
- Evaluate generated samples (KID in classifier feature space + optional latent MMD):
      - scripts/mnist01_eval.py

### Typical commands

1) Data + encoding

```bash
conda activate qml_gpu

python scripts/mnist01_prepare.py --out data/mnist01
python scripts/mnist01_encode_states.py --data data/mnist01
python scripts/mnist01_train_classifier.py --data data/mnist01
```

2) QGAN

```bash
python scripts/mnist01_train_qgan.py --data data/mnist01 --out data/mnist01/models/qgan --n 8
python scripts/mnist01_generate_qgan.py --ckpt data/mnist01/models/qgan/qgan_mnist01_n8na0Lg40Lc12_cy3.npz --out data/mnist01/gen/qgan
python scripts/mnist01_eval.py --data data/mnist01 --gen data/mnist01/gen/qgan
```

3) QDT

```bash
python scripts/mnist01_train_qdt.py --data data/mnist01 --out data/mnist01/models/qdt --n 8
python scripts/mnist01_generate_qdt.py --ckpt data/mnist01/models/qdt/qdt_mnist01_n8na0L80_b128_e20000.npz --out data/mnist01/gen/qdt
python scripts/mnist01_eval.py --data data/mnist01 --gen data/mnist01/gen/qdt
```

4) QDDPM

```bash
python scripts/mnist01_make_diffusion_qddpm.py --data data/mnist01 --out data/mnist01/diff --n 8 --T 20 --N 5000
python scripts/mnist01_train_qddpm.py --data data/mnist01 --out data/mnist01/models/qddpm --n 8 --na 2 --T 20 --L 6 --N_train 256 --epochs 3000 --resume
python scripts/mnist01_generate_qddpm.py --ckpt data/mnist01/models/qddpm/qddpm_mnist01_n8na2T20L6.npz --out data/mnist01/gen/qddpm --n 8 --na 2 --T 20 --L 6
python scripts/mnist01_eval.py --data data/mnist01 --gen data/mnist01/gen/qddpm
```
