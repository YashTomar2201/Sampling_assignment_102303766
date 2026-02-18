# Sampling Assignment - Credit Card Fraud Detection

This assignment explores how different sampling techniques affect the performance of machine learning models on an imbalanced dataset (credit card transactions).

---

## Objective

The dataset used here is highly imbalanced — most transactions are legitimate, with only a small fraction being fraudulent. The goal is to:

- Balance the dataset using SMOTE
- Create 5 samples using 5 different sampling techniques
- Train 5 ML models on each sample
- Compare accuracies and find which sampling technique works best for each model

---

## Dataset

Downloaded from:
[https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv)

The dataset contains credit card transaction records with a `Class` column as the target (0 = legitimate, 1 = fraud).

---

## How to Run

1. Clone this repo
```bash
git clone <your-repo-link>
cd <your-repo-folder>
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

3. Run the script
```bash
python main.py
```

The script will automatically download the dataset from GitHub. If the download fails for any reason, place `Creditcard_data.csv` in the same folder and run again.

---

## What the Script Does

**Step 1 - Load Data**
Reads the CSV directly from the GitHub raw URL and saves a local copy.

**Step 2 - Balance the Dataset**
Uses SMOTE (Synthetic Minority Oversampling Technique) to fix the class imbalance. Falls back to manual random oversampling if `imbalanced-learn` isn't installed.

**Step 3 - Create 5 Samples**

| Sample | Technique | Description |
|--------|-----------|-------------|
| Sampling1 | Simple Random Sampling | Randomly picks rows without replacement |
| Sampling2 | Systematic Sampling | Picks every kth row from the dataset |
| Sampling3 | Stratified Sampling | Samples proportionally from each class |
| Sampling4 | Cluster Sampling | Groups data into 10 clusters, samples from each |
| Sampling5 | Bootstrap Sampling | Random sampling with replacement |

**Step 4 - Train 5 ML Models on Each Sample**

| Model | Algorithm |
|-------|-----------|
| M1 | Logistic Regression |
| M2 | Decision Tree |
| M3 | Random Forest |
| M4 | Support Vector Machine (RBF kernel) |
| M5 | K-Nearest Neighbors |

Each model is trained on an 80/20 train-test split with StandardScaler applied.

---

## Results

Accuracy table (%) across all model-sampling combinations:

|  | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|--|-----------|-----------|-----------|-----------|-----------|
| M1_LogReg  | - | - | - | - | - |
| M2_DTree   | - | - | - | - | - |
| M3_RForest | - | - | - | - | - |
| M4_SVM     | - | - | - | - | - |
| M5_KNN     | - | - | - | - | - |

> Fill in actual values after running the script. Results are also saved automatically to `sampling_results.csv`.

**Discussion**

- Random Forest (M3) generally tends to perform well across most sampling strategies due to its ensemble nature and robustness to noise.
- Sampling3 (Stratified) usually gives more stable results since it preserves the class distribution during sampling.
- Bootstrap sampling can sometimes introduce duplicates which inflates training accuracy — worth keeping in mind when comparing.
- SVM (M4) and Logistic Regression (M1) are sensitive to the sample composition, so their results vary more across sampling techniques.

---

## Output Files

| File | Description |
|------|-------------|
| `sampling_results.csv` | Accuracy table saved as CSV |
| `sampling_heatmap.png` | Heatmap visualizing model vs sampling accuracy |
| `Creditcard_data.csv` | Local copy of the dataset |

---

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
