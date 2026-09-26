# Week 5 - Comprehensive Data Science Project
# Integrates data acquisition/EDA, visualization, hypothesis testing,
# and machine-learning model development using the UCI Wine dataset.

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
wine = pd.read_csv("wine_dataset.csv")
features = [c for c in wine.columns if c not in ["target", "class_name"]]
X = wine[features]
y = wine["target"]

# Basic quality checks
print("Shape:", wine.shape)
print("Missing values:", wine.isna().sum().sum())
print("Duplicates:", wine.duplicated().sum())
print("Class counts:\n", wine["class_name"].value_counts())

# Descriptive analysis
print("\nClass means:\n", wine.groupby("class_name")[features].mean())
print("\nCorrelation matrix:\n", X.corr())

# Hypothesis-oriented comparison: alcohol by class
print("\nAlcohol means by class:\n", wine.groupby("class_name")["alcohol"].mean())

# Model development
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, average="macro"))
    print("Recall:", recall_score(y_test, pred, average="macro"))
    print("F1:", f1_score(y_test, pred, average="macro"))
    print("ROC-AUC:", roc_auc_score(y_test, proba, multi_class="ovr"))

# Five-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name} CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

# Permutation importance
logit = models["Logistic Regression"]
logit.fit(X_train, y_train)
result = permutation_importance(logit, X_test, y_test, n_repeats=20, random_state=42)
importance = pd.Series(result.importances_mean, index=features).sort_values(ascending=False)
print("\nPermutation importance:\n", importance)

# PCA for a compact visual story
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
plt.scatter(coords[:, 0], coords[:, 1], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA View of Wine Data")
plt.show()
