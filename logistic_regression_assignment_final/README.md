# 🌸 Assignment 01 — Logistic Regression Classification

**Roll No:** su92-bsaim-s23-015  
**Subject:** Machine Learning  
**Dataset:** Iris Flower Dataset (CSV/Tabular)

---

##  Overview

This project implements **Logistic Regression** to classify Iris flowers into three species:
- Setosa
- Versicolor
- Virginica

---

##  Project Structure

```
logistic_regression_assignment/
├── model/
│   ├── train_model.py       # Training, evaluation, plots
│   └── plots/               # Generated after training
│       ├── confusion_matrix.png
│       └── feature_distributions.png
├── ui/
│   └── app.py               # Streamlit UI
├── saved_model/             # Generated after training
│   ├── logistic_model.pkl
│   └── scaler.pkl
├── requirements.txt
└── README.md
```

---

##  How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the model
```bash
cd model
python train_model.py
```

### Step 3: Launch the UI
```bash
cd ui
streamlit run app.py
```

---

##  Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~97%   |
| Precision | ~97%   |
| Recall    | ~97%   |
| F1-Score  | ~97%   |

---

##  Preprocessing Steps

1. **Missing value handling** — filled with column mean (none in Iris)
2. **Feature scaling** — StandardScaler normalization
3. **Train/Test split** — 80% train, 20% test (stratified)

---
##  Screenshots

### Confusion Matrix
![Confusion Matrix](logistic_regression_assignment_final/model/plots/confusion_matrix.png?raw=true)

![Feature Distributions](logistic_regression_assignment_final/model/plots/feature_distributions.png?raw=true)
---

##  Tech Stack

- Python 3.10+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- Streamlit (UI)
