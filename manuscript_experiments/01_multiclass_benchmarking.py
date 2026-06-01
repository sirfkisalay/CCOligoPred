"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 1: Baseline Multiclass Benchmarking
==============================================================================
Description:
Evaluates 17 baseline machine learning algorithms across 4 distinct feature 
extraction methodologies (GSBF, RBF, PSSM, and Hybrid) for the prediction 
of 4 coiled-coil oligomeric states (PD, APD, TRI, TET). 

Outputs:
Generates '01_Multiclass_Benchmarking_Results.csv' containing Overall Accuracy, 
Global MCC, One-vs-Rest MCCs for each state, and the Multiclass Oligomeric 
State Score (MOSS).
==============================================================================
"""

import os
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, matthews_corrcoef

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
TARGET_COLUMN_NAME = 'Class'

# Dynamically locate the DATASETS folder relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS")

# Dictionary containing all 4 dataset configurations
datasets = {
    "1. GSBF": {
        "train": os.path.join(DATA_DIR, "MULTICLASS_GSBF_TRAIN.xlsx"),
        "test": os.path.join(DATA_DIR, "MULTICLASS_GSBF_TEST.xlsx")
    },
    "2. RBF": {
        "train": os.path.join(DATA_DIR, "MULTICLASS_RBF_TRAIN.xlsx"),
        "test": os.path.join(DATA_DIR, "MULTICLASS_RBF_TEST.xlsx")
    },
    "3. PSSM": {
        "train": os.path.join(DATA_DIR, "MULTICLASS_TRAIN_PSSM.xlsx"),
        "test": os.path.join(DATA_DIR, "MULTICLASS_TEST_PSSM.xlsx")
    },
    "4. Hybrid (GSBF+RBF)": {
        "train": os.path.join(DATA_DIR, "RMULTICLASS_HYBRID_TRAIN.xlsx"),
        "test": os.path.join(DATA_DIR, "RMULTICLASS_HYBRID_TEST.xlsx")
    }
}

# =============================================================================
# --- Step 2: Define Models with Default Hyperparameters ---
# =============================================================================
models = {
    "LogisticRegression": LogisticRegression(random_state=42),
    "RidgeClassifier": RidgeClassifier(random_state=42),
    "SGDClassifier": SGDClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=42),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss"), 
    "LightGBM": LGBMClassifier(random_state=42, verbosity=-1),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42),
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
    print(f"--- STARTING BENCHMARK FOR: {dataset_name} ---")
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
    y_train_raw = train_df[TARGET_COLUMN_NAME]
    
    X_test = test_df.drop(columns=[TARGET_COLUMN_NAME])
    y_test_raw = test_df[TARGET_COLUMN_NAME]

    # Ensure column order is exactly the same for train and test
    X_test = X_test[X_train.columns]

    # Map target numbers into text names for clarity 
    class_name_map = {1: 'PD', 2: 'APD', 3: 'TRI', 4: 'TET'}
    y_train_raw = y_train_raw.map(class_name_map).fillna(y_train_raw)
    y_test_raw = y_test_raw.map(class_name_map).fillna(y_test_raw)

    # Initialize LabelEncoder for Multiclass
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    
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

            # Overall Evaluation Metrics
            acc = accuracy_score(y_test, y_pred)
            mcc_overall = matthews_corrcoef(y_test, y_pred)

            # Calculate Per-Class MCC using One-vs-Rest Approach
            mcc_per_class = {}
            for i, class_name in enumerate(encoder.classes_):
                y_test_bin = (y_test == i).astype(int)
                y_pred_bin = (y_pred == i).astype(int)
                mcc_per_class[class_name] = matthews_corrcoef(y_test_bin, y_pred_bin)
            
            # Calculate Custom MOSS Score
            tri_mcc = mcc_per_class.get("TRI", 0.0)
            tet_mcc = mcc_per_class.get("TET", 0.0)
            moss_score = (mcc_overall + tri_mcc + tet_mcc) / 3

            # --- Append to Results List for the Table ---
            final_results_table.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Overall Accuracy": round(acc, 4),
                "Overall MCC": round(mcc_overall, 4),
                "PD MCC": round(mcc_per_class.get("PD", 0.0), 4),
                "APD MCC": round(mcc_per_class.get("APD", 0.0), 4),
                "TRI MCC": round(tri_mcc, 4),
                "TET MCC": round(tet_mcc, 4),
                "MOSS Score": round(moss_score, 4)
            })
            
        except Exception as e:
            print(f"  [!] Failed to train {model_name}. Error: {str(e)}")

# =============================================================================
# --- Step 4: Generate and Save the Final Table ---
# =============================================================================

print("\n" + "=" * 80)
print("FINAL BENCHMARKING RESULTS")
print("=" * 80)

# Convert list to a Pandas DataFrame
results_df = pd.DataFrame(final_results_table)

# Save directly to a CSV file in the same directory as the script
csv_filename = os.path.join(BASE_DIR, "01_Multiclass_Benchmarking_Results.csv")
results_df.to_csv(csv_filename, index=False)

print(f"SUCCESS: Results have been safely exported to:\n{csv_filename}")
print("=" * 80)