import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

def train_overconfidence_classifier(
    csv_file="overconfidence_dataset.csv", 
    model_out="overconfidence_classifier.pkl",
    do_crossval=False
):
    df = pd.read_csv(csv_file)

    feature_cols = ["Confidence", "Uncertainty", "Prediction", "Dataset", "Model", "Method"]
    X_raw = df[feature_cols]
    y = df["Overconfident"].values

    categorical_features = ["Prediction", "Dataset", "Model", "Method"] 
    numeric_features = ["Confidence", "Uncertainty"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough"
    )

    base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   base_clf)
    ])

    if do_crossval:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        acc_scores, f1_scores = [], []

        for train_idx, val_idx in kf.split(X_raw):
            X_train_fold, X_val_fold = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            pipeline.fit(X_train_fold, y_train_fold)
            y_val_pred = pipeline.predict(X_val_fold)

            acc_scores.append(accuracy_score(y_val_fold, y_val_pred))
            f1_scores.append(f1_score(y_val_fold, y_val_pred))

        print(f"K-Fold CV Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
        print(f"K-Fold CV F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Final Hold-Out Accuracy: {acc:.4f}")
    print(f"Final Hold-Out F1 Score: {f1:.4f}")

    joblib.dump(pipeline, model_out)
    print(f"Overconfidence classifier saved as '{model_out}'")

if __name__ == "__main__":
    train_overconfidence_classifier(
        csv_file="overconfidence_dataset.csv",
        model_out="overconfidence_classifier.pkl",
        do_crossval=True  
    )
