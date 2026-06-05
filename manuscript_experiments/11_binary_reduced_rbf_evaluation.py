"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 11: Binary Reduced RBF Evaluation Protocol (Reduced Feature Sets)
==============================================================================
Description:
Evaluates the cross-performance of the 4 optimized baseline models (RandomForest, 
XGBoost, LightGBM, GradientBoosting) against the 3 reduced feature subsets 
extracted in Step 10 (ANOVA, LGBM Importance, RFECV) for the binary task 
(Target: Trimer vs. Rest).

Methodology:
- Feature Subsetting: Dynamic extraction based on Step 10 JSON outputs.
- Pipeline: Impute -> Scale -> SMOTE -> Evaluate (Leakage-proof via Imblearn).
- Target Metric: Global MCC, Per-Class MCC (One-vs-Rest).
- Outputs: '11_Binary_Evaluation_Results.csv'
==============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
import warnings

# Sklearn & Metrics
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

# Resampling & Pipeline (CRITICAL for No Leakage)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline 

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
# Dynamically locate the DATASETS/binary folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS", "binary")

TRAIN_FILE = "BINARY_RBF_TRAIN.xlsx"
TEST_FILE  = "BINARY_RBF_TEST.xlsx"
TARGET_COLUMN = "State_tri"
CLASS_NAMES = ['Rest', 'Tri'] # 0, 1
RANDOM_STATE = 42

# Feature Set Filenames (From Step 10)
FEATURE_FILES = {
    "Binary_ANOVA": "Binary_ANOVA_Features.json",
    "Binary_LGBM":  "Binary_LGBM_Importance_Features.json",
    "Binary_RFECV": "Binary_RFECV_Features.json"
}

# --- OPTIMIZED HYPERPARAMETERS ---
rf_params = {
    'n_estimators': 1300, 'max_depth': 23, 'min_samples_leaf': 9, 'max_features': 0.5,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': 0
}

xgb_params = {
    'n_estimators': 1150, 'max_depth': 12, 'learning_rate': 0.0405246, 
    'subsample': 0.75433, 'colsample_bytree': 0.60561,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0,
    'objective': 'binary:logistic', 'eval_metric': 'logloss' 
}

lgb_params = {
    'n_estimators': 860, 'learning_rate': 0.046423, 'num_leaves': 30, 'max_depth': 13,
    'subsample': 0.80635, 'colsample_bytree': 0.67299, 
    'reg_alpha': 1.1078e-08, 'reg_lambda': 0.17206,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1,
    'objective': 'binary' 
}

gb_params = {
    'n_estimators': 1350, 'learning_rate': 0.021942, 'max_depth': 9, 
    'min_samples_leaf': 8, 'subsample': 0.96444, 'max_features': None,
    'random_state': RANDOM_STATE
}

# Model Dictionary
models_to_train = {
    "RandomForest": RandomForestClassifier(**rf_params),
    "XGBoost": xgb.XGBClassifier(**xgb_params),
    "LightGBM": lgb.LGBMClassifier(**lgb_params),
    "GradientBoosting": GradientBoostingClassifier(**gb_params)
}

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def load_feature_list(filepath):
    """Safely loads feature list from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_per_class_mcc(y_true, y_pred, classes):
    """Calculates MCC for each class (One-vs-Rest approach)."""
    cm = confusion_matrix(y_true, y_pred)
    res = {}
    for i, cls_name in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = (tp * tn - fp * fn) / denom if denom != 0 else 0.0
        res[cls_name] = mcc
    return res

# ==============================================================================
# 3. DATA LOADING
# ==============================================================================
print(f"[*] Loading datasets from {DATA_DIR}...")
train_df = pd.read_excel(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_excel(os.path.join(DATA_DIR, TEST_FILE))

# Prepare X and Y
y_train_full = train_df[TARGET_COLUMN].astype(int)
y_test_full  = test_df[TARGET_COLUMN].astype(int)

# SMOTE Neighbor calculation (Dynamic based on smallest class)
min_class_size = Counter(y_train_full).most_common()[-1][1]
k_neigh = min(5, max(1, min_class_size - 1))
print(f" -> Data Loaded. Minority Class Size: {min_class_size}. SMOTE k_neighbors={k_neigh}\n")

# ==============================================================================
# 4. MAIN EVALUATION LOOP
# ==============================================================================
results_log = []

print("="*90)
print(f"{'FEATURE SET':<20} | {'MODEL':<18} | {'G-MCC':<8} | {'PER CLASS MCC (Rest, Tri)'}")
print("="*90)

for feat_name, json_file in FEATURE_FILES.items():
    
    # 1. Load Feature Names
    json_path = os.path.join(DATA_DIR, json_file)
        
    try:
        selected_feats = load_feature_list(json_path)
    except FileNotFoundError:
        print(f"[!] Error: Could not find {json_file}. Did you run Step 10? Skipping...")
        continue

    # 2. Subset Data (Strictly based on selected features)
    valid_feats = [f for f in selected_feats if f in train_df.columns]
    
    X_train_sub = train_df[valid_feats]
    X_test_sub  = test_df[valid_feats]

    for model_name, model_inst in models_to_train.items():
        
        # 3. Construct Pipeline (Leakage Proof)
        # Order: Impute -> Scale -> SMOTE (Train only) -> Classifier
        pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neigh)),
            ('clf', clone(model_inst)) # Clone to reset model for each run
        ])
        
        # 4. Train
        pipeline.fit(X_train_sub, y_train_full)
        
        # 5. Predict (Pipeline automatically applies scaling/imputation to test, but NOT SMOTE)
        y_pred = pipeline.predict(X_test_sub)
        
        # 6. Evaluate
        global_mcc = matthews_corrcoef(y_test_full, y_pred)
        pc_mcc = calculate_per_class_mcc(y_test_full, y_pred, CLASS_NAMES)
        
        # Log Result
        res_entry = {
            'Feature_Method': feat_name,
            'Model': model_name,
            'Num_Features': len(valid_feats),
            'Global_MCC': global_mcc,
            **{f"MCC_{k}": v for k, v in pc_mcc.items()}
        }
        results_log.append(res_entry)
        
        # Print Row
        pc_str = ", ".join([f"{v:.3f}" for v in pc_mcc.values()])
        print(f"{feat_name:<20} | {model_name:<18} | {global_mcc:.4f}   | [{pc_str}]")

print("="*90)

# ==============================================================================
# 5. SAVE FINAL RESULTS
# ==============================================================================
results_df = pd.DataFrame(results_log)
out_file = os.path.join(BASE_DIR, "11_Binary_Evaluation_Results.csv")
results_df.to_csv(out_file, index=False)
print(f"\n✅ Detailed results successfully saved to:\n{out_file}")

# Identify Best Combination
best_run = results_df.loc[results_df['Global_MCC'].idxmax()]
print("\n🏆 BEST PERFORMING COMBINATION (Binary Task) 🏆")
print(f"Feature Set : {best_run['Feature_Method']} ({best_run['Num_Features']} Features)")
print(f"Model       : {best_run['Model']}")
print(f"Global MCC  : {best_run['Global_MCC']:.4f}")
print(f"Tri MCC     : {best_run['MCC_Tri']:.4f}")
print(f"Rest MCC    : {best_run['MCC_Rest']:.4f}")