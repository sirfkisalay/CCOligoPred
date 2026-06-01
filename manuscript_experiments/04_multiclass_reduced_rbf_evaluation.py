"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 4: Multiclass Evaluation Protocol (Reduced Feature Sets)
==============================================================================
Description:
Evaluates the cross-performance of the 4 optimized baseline models (XGBoost, 
LightGBM, GradientBoosting, ExtraTrees) against the 6 reduced feature subsets 
extracted in Step 3 (ANOVA, RFECV, and Tree-Based Importances).

Methodology:
- Feature Subsetting: Dynamic extraction based on Step 3 JSON outputs.
- Pipeline: Impute -> Scale -> Evaluate (NO SMOTE).
- Target Metric: Global MCC, Per-Class MCC (One-vs-Rest), and MOSS Score.
- Outputs: '04_Multiclass_Evaluation_Results.csv'
==============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import warnings

# Sklearn & Metrics
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# Models
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
# Dynamically locate the DATASETS folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS")

TRAIN_FILE = "MULTICLASS_RBF_TRAIN.xlsx"
TEST_FILE  = "MULTICLASS_RBF_TEST.xlsx"

TARGET_COLUMN = "Class" 
CLASS_NAMES = ['PD', 'APD', 'TRI', 'TET'] 
RANDOM_STATE = 42

# Multiclass Feature Set Filenames (Generated from Step 3)
FEATURE_FILES = {
    "Multi_RFECV": "Multiclass_RFECV_Features_Raw.json",
    "Multi_ANOVA": "Multiclass_ANOVA_Features_Raw.json",
    "Multi_ExtraTrees": "Multiclass_ExtraTrees_Importance_Features_Raw.json",
    "Multi_GB": "Multiclass_GB_Importance_Features_mean_Raw.json",
    "Multi_XGB": "Multiclass_XGB_Importance_Features_mean_Raw.json",
    "Multi_LGBM": "Multiclass_LGBM_Importance_Features_mean_Raw.json"
}

# --- OPTIMIZED HYPERPARAMETERS ---
xgb_params = {
    'n_estimators': 700, 'max_depth': 11, 'learning_rate': 0.0147951583299575, 
    'subsample': 0.7776832317133678, 'colsample_bytree': 0.5870074173438061, 
    'gamma': 0.9200227958551773, 'min_child_weight': 4,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0,
    'objective': 'multi:softmax', 'eval_metric': 'mlogloss' 
}

lgb_params = {
    'n_estimators': 1250, 'max_depth': 23, 'learning_rate': 0.04472528989077509, 
    'num_leaves': 113, 'subsample': 0.9786298784381904, 'colsample_bytree': 0.8115594723131374, 
    'min_child_samples': 20, 'class_weight': 'balanced',
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1,
    'objective': 'multiclass', 'num_class': 4 
}

gb_params = {
    'n_estimators': 1900, 'learning_rate': 0.18698360549487777, 'max_depth': 9, 
    'min_samples_split': 20, 'min_samples_leaf': 15, 'subsample': 0.8386502590161441, 
    'max_features': 'log2',
    'random_state': RANDOM_STATE
}

et_params = {
    'n_estimators': 600, 'max_depth': 50, 'min_samples_split': 15, 
    'min_samples_leaf': 2, 'max_features': None, 'bootstrap': True, 
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': 0
}

# Model Dictionary
models_to_train = {
    "XGBoost": xgb.XGBClassifier(**xgb_params),
    "LightGBM": lgb.LGBMClassifier(**lgb_params),
    "GradientBoosting": GradientBoostingClassifier(**gb_params),
    "ExtraTrees": ExtraTreesClassifier(**et_params)
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
# 3. DATA LOADING & ENCODING
# ==============================================================================
print(f"[*] Loading datasets from: {DATA_DIR}")
train_df = pd.read_excel(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_excel(os.path.join(DATA_DIR, TEST_FILE))

# Prepare X and Y
y_train_raw = train_df[TARGET_COLUMN]
y_test_raw  = test_df[TARGET_COLUMN]

# Map targets consistently
class_name_map = {1: 'PD', 2: 'APD', 3: 'TRI', 4: 'TET'}
y_train_mapped = y_train_raw.map(class_name_map).fillna(y_train_raw)
y_test_mapped = y_test_raw.map(class_name_map).fillna(y_test_raw)

# Encode target labels (Maps PD, APD, TRI, TET to 0, 1, 2, 3)
le = LabelEncoder()
y_train_full = le.fit_transform(y_train_mapped)
y_test_full = le.transform(y_test_mapped)

print(f" -> Target Labels mapped as: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f" -> Classes found: {len(np.unique(y_train_full))}. Training samples: {len(y_train_full)}")

# ==============================================================================
# 4. MAIN EVALUATION LOOP
# ==============================================================================
results_log = []

print("\n" + "="*115)
print(f"{'FEATURE SET':<20} | {'MODEL':<18} | {'G-MCC':<8} | {'MOSS':<8} | {'PER CLASS MCC ' + str(CLASS_NAMES)}")
print("="*115)

for feat_name, json_file in FEATURE_FILES.items():
    
    # 1. Load Feature Names
    json_path = os.path.join(DATA_DIR, json_file)
        
    try:
        selected_feats = load_feature_list(json_path)
    except FileNotFoundError:
        print(f"[!] Error: Could not find {json_file}. Did you run Step 3? Skipping...")
        continue

    # 2. Subset Data (Strictly based on selected features)
    valid_feats = [f for f in selected_feats if f in train_df.columns]
    X_train_sub = train_df[valid_feats]
    X_test_sub  = test_df[valid_feats]

    for model_name, model_inst in models_to_train.items():
        
        # 3. Construct Standard Pipeline (Leakage Proof - NO SMOTE)
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', clone(model_inst)) 
        ])
        
        # 4. Train & Predict
        pipeline.fit(X_train_sub, y_train_full)
        y_pred = pipeline.predict(X_test_sub)
        
        # 5. Evaluate Metrics
        global_mcc = matthews_corrcoef(y_test_full, y_pred)
        pc_mcc = calculate_per_class_mcc(y_test_full, y_pred, CLASS_NAMES)
        
        moss_score = (global_mcc + pc_mcc.get("TRI", 0.0) + pc_mcc.get("TET", 0.0)) / 3
        
        # Log Result
        res_entry = {
            'Feature_Method': feat_name,
            'Model': model_name,
            'Num_Features': len(valid_feats),
            'Global_MCC': global_mcc,
            'MOSS_Score': moss_score,
            **{f"MCC_{k}": v for k, v in pc_mcc.items()}
        }
        results_log.append(res_entry)
        
        # Print Row
        pc_str = ", ".join([f"{v:.3f}" for v in pc_mcc.values()])
        print(f"{feat_name:<20} | {model_name:<18} | {global_mcc:.4f}   | {moss_score:.4f}   | [{pc_str}]")

print("="*115)

# ==============================================================================
# 5. SAVE FINAL RESULTS
# ==============================================================================
results_df = pd.DataFrame(results_log)
out_file = os.path.join(BASE_DIR, "04_Multiclass_Evaluation_Results_NoSmote.csv")
results_df.to_csv(out_file, index=False)
print(f"\nDetailed results saved to: {out_file}")

# Identify Best Combination by MOSS Score
best_run = results_df.loc[results_df['MOSS_Score'].idxmax()]
print("\n🏆 BEST PERFORMING CONFIGURATION (By MOSS Score) 🏆")
print(f"Feature Set : {best_run['Feature_Method']}")
if 'Resampler' in best_run:
    print(f"Resampler   : {best_run['Resampler']}")
print(f"Model       : {best_run['Model']}")
print("-" * 35)
print(f"Global MCC  : {best_run['Global_MCC']:.4f}")
print(f"MOSS Score  : {best_run['MOSS_Score']:.4f}")
print("-" * 35)
print(f"PD MCC      : {best_run['MCC_PD']:.4f}")
print(f"APD MCC     : {best_run['MCC_APD']:.4f}")
print(f"TRI MCC     : {best_run['MCC_TRI']:.4f}")
print(f"TET MCC     : {best_run['MCC_TET']:.4f}")