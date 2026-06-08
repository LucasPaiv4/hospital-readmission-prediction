# Hospital Readmission Prediction

A machine learning project applied to a real clinical problem: predicting the
readmission of diabetic patients from hospital records. The work covers the full
pipeline — exploratory analysis, preprocessing, supervised and unsupervised
modeling, and a critical evaluation of the results.

**Author:** Lucas Paiva de Araújo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucasPaiv4/hospital-readmission-prediction/blob/main/Prova_paradigmas.ipynb)

## The problem

The goal is to predict **hospital readmission** of diabetic patients (readmitted
in under 30 days, over 30 days, or not readmitted) from demographic data,
admission history, diagnoses, and medication usage.

The dataset is the
[Diabetes 130-US hospitals](https://archive.ics.uci.edu/dataset/296/) set from
the UCI repository, loaded directly from its URL inside the notebook (no manual
download required).

## Analysis pipeline

1. **Exploratory analysis** — attribute types, missing values, duplicates, and
   the (imbalanced) distribution of the target variable.
2. **Preprocessing** — removal of identifier columns and type-aware imputation:
   median for numeric features (robust to outliers) and `"Unknown"` for
   categorical ones (preserving missingness as a potential clinical signal).
3. **Encoding** — comparison between Label Encoding and One-Hot Encoding, with
   stratified sampling to keep the computational cost manageable.
4. **Dimensionality reduction** — standardization and PCA retaining 95% of the
   variance.
5. **Supervised models** — Decision Tree, Naive Bayes, and MLP, each tuned over
   two hyperparameters via `GridSearchCV` with stratified cross-validation
   (10 folds), evaluated with the macro F1-score.
6. **Unsupervised models** — KMeans and hierarchical clustering (Ward and
   Complete), evaluated with Silhouette, Davies-Bouldin, and Calinski-Harabasz.
7. **Final reflection** — how the choices of imputation, encoding, and
   dimensionality reduction affected performance.

## Key findings

- Type-aware imputation preserves more information than a single uniform
  strategy, and the absence of certain exams can be a relevant clinical signal
  in itself.
- One-Hot Encoding avoids the artificial ordinality of Label Encoding but greatly
  increases dimensionality — which hurt GaussianNB and motivated the sampling.
- Across all three clustering metrics, the hierarchical methods showed
  seemingly superior scores, but collapsed nearly all points into a single
  cluster (a degenerate partition). KMeans, despite lower metrics, produced the
  most informative partition — expected behavior in high-dimensional data with
  many binary features.
- Applying PCA lowered the F1-score, indicating that the original features carry
  relevant, complementary signals; here, dimensionality reduction cost more than
  it gained.

## How to run

The easiest way is to open it in Google Colab via the badge above. To run
locally:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Prova_paradigmas.ipynb
```

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Jupyter
