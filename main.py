# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


# loading the dataset directly from github
url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"

try:
    df = pd.read_csv(url)
    print("dataset loaded, shape:", df.shape)
    df.to_csv("Creditcard_data.csv", index=False)
except:
    print("couldn't fetch from github, loading local copy instead")
    df = pd.read_csv("Creditcard_data.csv")

print(df.head())
print("\nclass counts before balancing:")
print(df['Class'].value_counts())


# separating features and label
X = df.drop(columns=['Class'])
y = df['Class']


# balancing the dataset using SMOTE
# the dataset is heavily imbalanced so we need this step
try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print("\nafter SMOTE:", Counter(y_res))

except ImportError:
    # if imblearn isn't installed, just do manual oversampling
    print("imblearn not found, doing manual oversampling")
    full = pd.concat([X, y], axis=1)
    maj = y.value_counts().idxmax()
    mn  = y.value_counts().idxmin()
    n   = y.value_counts()[maj]

    minority_upsampled = full[full['Class'] == mn].sample(n=n, replace=True, random_state=42)
    balanced = pd.concat([full[full['Class'] == maj], minority_upsampled]).reset_index(drop=True)

    X_res = balanced.drop(columns=['Class'])
    y_res = balanced['Class']
    print("after oversampling:", Counter(y_res))


# putting it back into a dataframe
df_bal = pd.concat([
    pd.DataFrame(X_res, columns=X.columns),
    pd.Series(y_res, name='Class')
], axis=1)

print("\nbalanced dataset shape:", df_bal.shape)


# creating 5 different samples using different sampling techniques

n_total = len(df_bal)
sample_size = min(500, n_total)

# simple random sampling - just randomly pick rows
s1 = df_bal.sample(n=sample_size, random_state=42).reset_index(drop=True)

# systematic sampling - pick every kth row
k = n_total // sample_size
indices = list(range(0, n_total, k))[:sample_size]
s2 = df_bal.iloc[indices].reset_index(drop=True)

# stratified sampling - make sure both classes are represented proportionally
s3, _ = train_test_split(df_bal, train_size=sample_size, stratify=df_bal['Class'], random_state=42)
s3 = s3.reset_index(drop=True)

# cluster sampling - divide data into clusters and sample from each
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df_bal['cluster'] = kmeans.fit_predict(X_res)
per_cluster = sample_size // 10
chunks = []
for i in range(10):
    chunk = df_bal[df_bal['cluster'] == i]
    chunks.append(chunk.sample(n=min(per_cluster, len(chunk)), random_state=42))
s4 = pd.concat(chunks).drop(columns=['cluster']).reset_index(drop=True)
df_bal.drop(columns=['cluster'], inplace=True)

# bootstrap sampling - sampling with replacement
s5 = df_bal.sample(n=sample_size, replace=True, random_state=42).reset_index(drop=True)

print("\nsampling done:")
print(f"  S1 (random):      {len(s1)} rows")
print(f"  S2 (systematic):  {len(s2)} rows")
print(f"  S3 (stratified):  {len(s3)} rows")
print(f"  S4 (cluster):     {len(s4)} rows")
print(f"  S5 (bootstrap):   {len(s5)} rows")


# defining the 5 models
models = {
    "M1_LogReg":  LogisticRegression(max_iter=1000, random_state=42),
    "M2_DTree":   DecisionTreeClassifier(random_state=42),
    "M3_RForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "M4_SVM":     SVC(kernel='rbf', random_state=42),
    "M5_KNN":     KNeighborsClassifier(n_neighbors=5),
}

samples = {
    "Sampling1": s1,
    "Sampling2": s2,
    "Sampling3": s3,
    "Sampling4": s4,
    "Sampling5": s5,
}

# training each model on each sample and storing the accuracy
results = {}

for m_name, model in models.items():
    results[m_name] = {}
    for s_name, sdf in samples.items():
        X_s = sdf.drop(columns=['Class'])
        y_s = sdf['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s, test_size=0.2, stratify=y_s, random_state=42
        )

        # scaling is important especially for SVM and LR
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)

        model.fit(X_train, y_train)
        acc = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)
        results[m_name][s_name] = acc


# showing the results
results_df = pd.DataFrame(results).T
results_df.index.name = "Model"

print("\n\nAccuracy table (%):")
print(results_df.to_string())

# which sampling technique worked best for each model
print("\n\nBest sampling per model:")
for model in results_df.index:
    best = results_df.loc[model].idxmax()
    best_acc = results_df.loc[model].max()
    print(f"  {model}: {best} ({best_acc}%)")

# which model performed best on each sampling technique
print("\nBest model per sampling technique:")
for col in results_df.columns:
    best = results_df[col].idxmax()
    best_acc = results_df[col].max()
    print(f"  {col}: {best} ({best_acc}%)")


results_df.to_csv("sampling_results.csv")
print("\nresults saved to sampling_results.csv")


# heatmap to visualize everything
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 5))
    sns.heatmap(results_df.astype(float), annot=True, fmt=".2f",
                cmap="Blues", linewidths=0.4, cbar_kws={"label": "Accuracy %"})
    plt.title("Model Accuracy across Sampling Techniques")
    plt.tight_layout()
    plt.savefig("sampling_heatmap.png", dpi=150)
    print("heatmap saved to sampling_heatmap.png")
    plt.show()

except ImportError:
    print("matplotlib not installed, skipping the heatmap")