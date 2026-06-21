# Generative Digit Classification & Missing Feature Reconstruction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Dataset: MNIST](https://img.shields.io/badge/Dataset-MNIST-orange.svg)](https://yann.lecun.com/exdb/mnist/)

A Python-based machine learning project implementing **Generative Classification**, **Expectation-Maximization (EM) for Gaussian Mixture Models (GMM)**, and **Probabilistic Image Imputation (Reconstruction)** on the MNIST dataset from scratch. 

This repository was developed as part of the **CS771 (Introduction to Machine Learning)** course at the **Indian Institute of Technology Kanpur (IITK)**.

---

## 📌 Project Overview

This project shifts focus from traditional discriminative models (like SVMs or Neural Networks) to **probabilistic generative models**. Instead of learning boundaries between classes, generative modeling learns the underlying probability distribution of the data: $P(X \mid Y)$.

### Key Features
1. **Gaussian Mixture Models (GMM)**: Implemented the **Expectation-Maximization (EM)** algorithm from scratch to model multi-modal digit distributions (e.g., thin vs. bold zeros, upright vs. slanted fours).
2. **Covariance Analysis**: Explored how capturing full covariance matrices (compared to isotropic/diagonal covariance) preserves relationships between pixels—such as anti-correlations in upright vs. inverted 7s.
3. **Generative Classifier**: Achieved **85.7% accuracy** on MNIST classification by fitting a multivariate Gaussian per digit class.
4. **Image Imputation (Restoration)**: Reconstructed censored images (where the central $21\%$ of pixels were wiped out) using **Conditional Gaussian Distributions**, successfully restoring missing structural details.

---

## 🧮 Mathematical Concepts Implemented

### 1. Expectation-Maximization (EM) for GMMs
To fit a mixture of $C$ Gaussians without explicit style labels, we treat the component assignments as latent variables and optimize the expected log-likelihood iteratively:
* **E-Step**: Compute the soft-responsibilities (posterior probabilities) for each data point $x^i$ belonging to cluster $c$:
  $$q_c^i = P(z^i = c \mid x^i, \theta) \propto \exp\left(-\frac{1}{2} \|x^i - \mu^c\|^2_2\right)$$
* **M-Step**: Update the cluster means $\mu^c$ and covariance matrices $\Sigma^c$ via weighted Maximum Likelihood Estimation (MLE):
  $$\mu^c = \frac{\sum_i q_c^i x^i}{\sum_i q_c^i}$$
  $$\Sigma^c = \frac{1}{\sum_i q_c^i} \sum_i q_c^i (x^i - \mu^c)(x^i - \mu^c)^T$$

### 2. Missing Feature Imputation via Conditional Gaussians
When a portion of the test image features is censored (represented by $X_{\text{unobs}}$) while others are observed ($X_{\text{obs}}$), we partition the learned mean and covariance matrix:
$$\mu = \begin{bmatrix} \mu_{\text{obs}} \\ \mu_{\text{unobs}} \end{bmatrix}, \quad \Sigma = \begin{bmatrix} \Sigma_{\text{obs, obs}} & \Sigma_{\text{obs, unobs}} \\ \Sigma_{\text{unobs, obs}} & \Sigma_{\text{unobs, unobs}} \end{bmatrix}$$

Using the properties of multivariate Gaussians, we compute the conditional distribution $P(X_{\text{unobs}} \mid X_{\text{obs}})$ to predict the missing pixels:
$$\mu_{\text{unobs} \mid \text{obs}} = \mu_{\text{unobs}} + \Sigma_{\text{unobs, obs}} \Sigma_{\text{obs, obs}}^{-1} (X_{\text{obs}} - \mu_{\text{obs}})$$

---

## 📂 Repository Structure

* `lec15.ipynb`: Interactive Jupyter Notebook containing the walk-through of Gaussians, EM algorithms, classifiers, and imputation experiments.
* `cs771/`: Python helper package.
  * [plotData.py](file:///c:/Users/kondr/Desktop/CS771_notes/project/cs771/plotData.py): Visualization utilities for plotting 2D classifiers, training boundaries, and MNIST grids.
  * [utils.py](file:///c:/Users/kondr/Desktop/CS771_notes/project/cs771/utils.py): High-performance utility functions (e.g., vectorized pairwise distance calculations).
* `.gitignore`: Configured to exclude system caches, Jupyter checkpoints, and the large raw dataset directory.

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have Python 3.8+ installed along with the following libraries:
```bash
pip install numpy matplotlib jupyter
```

### 📥 Dataset Setup
1. Download the MNIST dataset from Kaggle: [Read MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data).
2. Extract the files and structure your project directory as follows:
   ```
   ├── cs771/
   │   ├── __init__.py
   │   ├── plotData.py
   │   └── utils.py
   ├── mnist/
   │   ├── t10k-images-idx3-ubyte/
   │   ├── t10k-labels-idx1-ubyte/
   │   ├── train-images-idx3-ubyte/
   │   └── train-labels-idx1-ubyte/
   ├── lec15.ipynb
   └── README.md
   ```

### 💻 Running the Experiments
Launch the Jupyter notebook interface:
```bash
jupyter notebook
```
Open `lec15.ipynb` and run the cells sequentially to visualize:
* Synthetic digit generation.
* Progression of the GMM EM algorithm.
* Image reconstruction comparisons (Censored vs. Imputed vs. Ground Truth).

---

## 📊 Results & Insights

* **Isotropic Gaussian Limitations**: A single spherical/isotropic Gaussian fails to capture digit diversity, generating extremely noisy, superimposed images.
* **GMM Representation Power**: Setting $C=10$ mixture components allows the model to cleanly segregate and generate distinct digit shapes and writing orientations entirely unsupervised.
* **Structural Recovery**: In the image reconstruction phase, even with $21\%$ of the core image censored, utilizing conditional covariance maps enables the model to resolve the missing geometry, yielding highly legible outputs and maintaining a classification accuracy of **78.6%**.

---

## 🎓 Acknowledgements
This project relies on helper utilities and code structures provided by the teaching staff of CS771, **CSE Department, IIT Kanpur**. The MNIST data loader utility is courtesy of Hojjat Khodabakhsh.
