# Sepsis Classification with FIS, ANFIS, and AutoML

This repository contains three projects developed for the **AIN421 Fuzzy Logic (Fall 2024)** course at Hacettepe University. The goal is to explore fuzzy logic-based and data-driven approaches to improve sepsis classification using real clinical data. Each phase expands upon the previous by incorporating different inference systems and automation strategies.

## 📁 Projects Overview

### 1. Fuzzy Inference System (FIS) - Mamdani-Type

- **Purpose:** Classify sepsis using interpretable, rule-based fuzzy logic.
- **Approach:** Manual definition of fuzzy sets and rules using physiological data.
- **Features Used:** Respiratory Rate, Heart Rate, BUN, Systolic BP, Bicarbonate
- **Libraries:** `scikit-fuzzy`, `scikit-learn`, `numpy`, `matplotlib`, `pandas`
- **Performance (Midterm):**
  - Accuracy: 71%
  - Recall (TPR): 82%
  - Precision: 67%
  - ROC AUC: 0.71

### 2. Adaptive Neuro-Fuzzy Inference System (ANFIS)

- **Purpose:** Learn fuzzy rules adaptively via neural networks.
- **Approach:** Use Gaussian membership functions and a modified ANFIS implementation.
- **Comparison Baseline:** Random Forest classifier
- **Libraries:** Custom ANFIS (adapted from [twmeggs/anfis](https://github.com/twmeggs/anfis)), `scikit-learn`, `numpy`, `matplotlib`
- **Performance (Final):**
  - ANFIS Accuracy: 89%, Recall: 87%, Precision: 91%, ROC AUC: 0.89
  - Random Forest Accuracy: 70%, Recall: 71%, ROC AUC: 0.70

### 3. AutoML Enhancement with Fuzzy Logic (PoC)

- **Objective:** Extend PyCaret to incorporate a Mamdani-type FIS component.
- **AutoML Platforms Evaluated:**
  - [`auto-sklearn`](https://github.com/automl/auto-sklearn)
  - [`MATLAB Fuzzy Logic Toolbox`](https://www.mathworks.com/products/fuzzy-logic.html)
  - [`PyCaret`](https://github.com/pycaret/pycaret) ← Selected
- **Extension Highlights:**
  - Integrates triangular membership functions and fuzzy rules.
  - Uses COA (Center of Area) defuzzification.
  - Supports hybrid reasoning for classification interpretation.

## 📊 Dataset

The dataset used for all models includes physiological data from septic and non-septic patients. For FIS, a sample of 500 records (250 per class) was used. For ANFIS and AutoML, larger subsets (1000–1500 samples) were used.

**Features:**
- `resp` (Respiratory Rate)
- `heart_rate` (Heart Rate)
- `bun` (Blood Urea Nitrogen)
- `bp_systolic` (Systolic Blood Pressure)
- `bicarbonate`

Target: `sepsis_icd` (binary classification: septic or non-septic)


## 🧪 Evaluation Metrics

* Confusion Matrix
* Accuracy, Precision, Recall (TPR), FPR, TNR, FNR
* F1 Score
* ROC AUC Score


## 📚 References

* Scikit-Fuzzy: [https://scikit-fuzzy.readthedocs.io/](https://scikit-fuzzy.readthedocs.io/)
* PyCaret: [https://pycaret.readthedocs.io/](https://pycaret.readthedocs.io/)
* ANFIS: [https://github.com/twmeggs/anfis](https://github.com/twmeggs/anfis)
* IEEE ANFIS Paper: [https://doi.org/10.1109/21.256541](https://doi.org/10.1109/21.256541)

