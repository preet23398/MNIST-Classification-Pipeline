# MNIST Classification using MLE, PCA, FDA, LDA & QDA

This project implements a classification pipeline from scratch using NumPy, applied on a filtered subset of the MNIST dataset (digits 0, 1, and 2). The pipeline includes:

- **Maximum Likelihood Estimation (MLE)**
- **Principal Component Analysis (PCA)**
- **Fisher’s Discriminant Analysis (FDA)**
- **Linear & Quadratic Discriminant Analysis (LDA & QDA)**

---

## Dataset & Structure

The MNIST dataset (IDX format) is used. Only digits `0`, `1`, and `2` are considered. From each class:
- 100 samples are used for **training**
- 100 samples are used for **testing**

---

## Outputs

Accuracy on training and testing data using:
- FDA + LDA/QDA (95% variance)
- PCA + LDA (95%)
- FDA + LDA/QDA (90%)
- FDA + LDA/QDA (first 2 PCA components)

Visualizations of transformed feature space using:
- PCA
- FDA

---

## Highlights

- Classification with only NumPy — no scikit-learn or ML libraries.
- PCA for unsupervised dimensionality reduction.
- FDA for supervised class separability.
- Evaluates both LDA and QDA under multiple dimension settings.
- Clean modular implementation, readable and extendable.
