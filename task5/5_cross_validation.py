"""Evaluate using cross-validation.
 - Runs stratified k-fold cross-validation for DecisionTree and RandomForest
 - Saves results to /mnt/data/cv_results.txt and prints summary
"""
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def load_prepare(path="/mnt/data/heart.csv"):
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    if "target" in df.columns:
        y = df["target"]; X = df.drop(columns=["target"])
    elif "output" in df.columns:
        y = df["output"]; X = df.drop(columns=["output"])
    else:
        y = df.iloc[:, -1]; X = df.iloc[:, :-1]
    return pd.get_dummies(X, drop_first=True), y

def main():
    X, y = load_prepare()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    dt_scores = cross_val_score(dt, X, y, cv=skf, scoring="accuracy")
    rf_scores = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
    out = "/mnt/data/cv_results.txt"
    with open(out, "w") as f:
        f.write("Decision Tree (max_depth=4) CV accuracies:\\n")
        f.write(str(dt_scores) + "\\n")
        f.write("mean: {:.4f}, std: {:.4f}\\n\\n".format(dt_scores.mean(), dt_scores.std()))
        f.write("Random Forest (200 trees) CV accuracies:\\n")
        f.write(str(rf_scores) + "\\n")
        f.write("mean: {:.4f}, std: {:.4f}\\n".format(rf_scores.mean(), rf_scores.std()))
    print("Decision Tree CV accuracies:", dt_scores, "mean:", dt_scores.mean(), "std:", dt_scores.std())
    print("Random Forest CV accuracies:", rf_scores, "mean:", rf_scores.mean(), "std:", rf_scores.std())
    print("Saved CV results to:", out)

if __name__ == "__main__":
    main()
