# 💼 Explainable AI and NLP-Powered Resume Screening System

## 🧠 Overview

This project builds a **fair, interpretable, and bias-aware AI-based resume screening system** using cutting-edge **NLP and machine learning** techniques. The goal is to support ethical and transparent hiring by reducing biases in automated resume evaluations.

---

## 🚀 Key Features

* **Data Preprocessing**: Standardizes resumes, extracts text features, and creates demographic signals (e.g., gender, race, age group).
* **Feature Extraction**: Uses **BERT embeddings** to capture rich semantic information from resume text.
* **Bias Detection**:

  * **Statistical Parity**: Checks equal selection rates across groups.
  * **Disparate Impact Ratio**: Assesses fairness between protected vs. unprotected groups.
  * **Intersectional Analysis**: Evaluates compound biases (e.g., gender + race).
* **Bias Mitigation**:

  * **Counterfactual Fairness**: Ensures predictions are stable across sensitive attribute changes.
  * **Adversarial Debiasing**: Trains fair embeddings by removing bias signals.
  * **Reweighing**: Adjusts sample weights to balance group representation.
* **Explainable AI (XAI)**:

  * Integrates **LIME** to highlight which resume sections influence decisions.

---

## ⚙️ Prerequisites

* **Python** 3.8+
* **Required Libraries**:

  * PyTorch
  * Transformers
  * Scikit-learn
  * Pandas
  * NumPy
  * Seaborn
  * Matplotlib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

