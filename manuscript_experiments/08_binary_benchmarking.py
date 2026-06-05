"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 8: Baseline Binary Benchmarking (GSBF, RBF, Hybrid)
==============================================================================
Description:
Evaluates 17 baseline machine learning algorithms across 3 distinct feature 
extraction methodologies (GSBF, RBF, and Hybrid) for the binary prediction 
task.

Methodology:
- Feature Sets: Automatically loops through GSBF, RBF, and Hybrid datasets.
- Preprocessing: Standard Scaling (Fit on Train, Transform on Test).
- Metrics: Accuracy, MCC, Sensitivity (Recall), and Specificity.
- Output: '08_Binary_Benchmarking_Results.csv'
==============================================================================
"""

import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

# Model Imports
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from imblearn.ensemble import BalancedRandomForestClassifier

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# =============================================================================
# --- Step 1: Dynamic Dataset Paths ---
# =============================================================================
# Update this if your actual column name differs across datasets
TARGET_COLUMN_NAME = 'State_tri' 

# Dynamically locate the binary DATASETS folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS", "binary")

# Dictionary containing all 3 dataset configurations
datasets = {
    "1. GSBF": {
        "train": os.path.join(DATA_DIR, "BINARY_GSBF_TRAIN.xlsx"),
        "test": os.path.join(DATA_DIR, "BINARY_GSBF_TEST.xlsx")
    },
    "2. RBF": {
        "train": os.path.join(DATA_DIR, "BINARY_RBF_TRAIN.xlsx"),
        "test": os.path.join(DATA_DIR, "BINARY_RBF_TEST.xlsx")
    },
    "3. Hybrid (GSBF+RBF)": {
        "train": os.path.join(DATA_DIR, "BINARY_HYBRID_TRAIN.xlsx"),
        "test": os.path.join(DATA_DIR, "BINARY_HYBRID_TEST.xlsx")
    }
}

# =============================================================================
# --- Step 2: Define Models with Original Configurations ---
# =============================================================================
models = {
    "LogisticRegression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42, class_weight='balanced'),
    "RidgeClassifier": RidgeClassifier(random_state=42, class_weight='balanced'),
    "SGDClassifier": SGDClassifier(random_state=42, class_weight='balanced', loss='log_loss'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, class_weight='balanced', verbosity=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42, class_weight='balanced'),
    "Bagging": BaggingClassifier(random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "BalancedRandomForest": BalancedRandomForestClassifier(random_state=42)
}

final_results_table = []

# =============================================================================
# --- Step 3: Loop Through Datasets and Benchmark ---
# =============================================================================

for dataset_name, paths in datasets.items():
    print("\n" + "#" * 80)
    print(f"--- STARTING BINARY BENCHMARK FOR: {dataset_name} ---")
    print("#" * 80)
    
    # Check if files exist before reading
    if not os.path.exists(paths["train"]) or not os.path.exists(paths["test"]):
        print(f"[!] ERROR: Files for {dataset_name} not found in {DATA_DIR}. Skipping.")
        continue

    # Load data
    train_df = pd.read_excel(paths["train"])
    test_df = pd.read_excel(paths["test"])

    # Prepare features (X) and labels (y)
    X_train = train_df.drop(columns=[TARGET_COLUMN_NAME])
    y_train = train_df[TARGET_COLUMN_NAME]
    
    X_test = test_df.drop(columns=[TARGET_COLUMN_NAME])
    y_test = test_df[TARGET_COLUMN_NAME]

    # Ensure column order is exactly the same for train and test
    X_test = X_test[X_train.columns]

    # Standardization (Fit on train, transform on test to prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and Evaluate Models
    for model_name, model in models.items():
        print(f"  -> Training {model_name}...")
        try:
            # Fit model and Predict
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Evaluation Metrics
            acc = accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Extract Confusion Matrix components (Assumes Binary: 0/1 or similar)
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                sens, spec = np.nan, np.nan

            # --- Append to Results List for the Table ---
            final_results_table.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": round(acc, 4),
                "MCC": round(mcc, 4),
                "Sensitivity": round(sens, 4),
                "Specificity": round(spec, 4)
            })
            
        except Exception as e:
            print(f"  [!] Failed to train {model_name}. Error: {str(e)}")

# =============================================================================
# --- Step 4: Generate and Save the Final Table ---
# =============================================================================

print("\n" + "=" * 80)
print("FINAL BINARY BENCHMARKING RESULTS")
print("=" * 80)

# Convert list to a Pandas DataFrame
results_df = pd.DataFrame(final_results_table)

# Save directly to a CSV file in the root manuscript_experiments directory
csv_filename = os.path.join(BASE_DIR, "08_Binary_Benchmarking_Results.csv")
results_df.to_csv(csv_filename, index=False)

print(f"SUCCESS: Results have been safely exported to:\n{csv_filename}")
print("=" * 80)