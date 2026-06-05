"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 13: Binary Final Threshold Optimization (Leakage-Proof OOF-CV)
==============================================================================
Description:
Performs Out-Of-Fold (OOF) cross-validation to derive optimal decision 
thresholds for the binary classification task (Target: State_tri).

Evaluates three pre-selected experimental branches:
- BRANCH A: Top 5 Models (Reduced Features, No Resampling)
- BRANCH B: Top 5 Models (Reduced Features, With Resampling)
- BRANCH C: Baseline (Full Features, No Resampling, All 4 Top Models)

Methodology:
- Probabilities generated via 5-Fold Stratified CV (Leakage-Proof via Imblearn).
- Thresholds optimized dynamically (0.05 to 0.95) to maximize Global MCC.
- Final test predictions explicitly evaluated at the optimized threshold.
==============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
import warnings

# Metrics & Selection
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

# Resampling
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
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
CLASS_NAMES = ['Rest', 'Tri']
RANDOM_STATE = 42
CV_FOLDS = 5  

# Feature Files Map (From Step 10)
FEAT_FILES = {
    "Binary_ANOVA": "Binary_ANOVA_Features.json",
    "Binary_RFECV": "Binary_RFECV_Features.json",
    "Binary_LGBM":  "Binary_LGBM_Importance_Features.json"
}

# --- OPTIMIZED HYPERPARAMETERS ---
rf_p = {'n_estimators': 1300, 'max_depth': 23, 'min_samples_leaf': 9, 'max_features': 0.5, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': 0}
xgb_p = {'n_estimators': 1150, 'max_depth': 12, 'learning_rate': 0.0405246, 'subsample': 0.75433, 'colsample_bytree': 0.60561, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}
lgb_p = {'n_estimators': 860, 'learning_rate': 0.046423, 'num_leaves': 30, 'max_depth': 13, 'subsample': 0.80635, 'colsample_bytree': 0.67299, 'reg_alpha': 1.1078e-08, 'reg_lambda': 0.17206, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1, 'objective': 'binary'}
gb_p = {'n_estimators': 1350, 'learning_rate': 0.02194282, 'max_depth': 9, 'min_samples_leaf': 8, 'subsample': 0.96444, 'max_features': None, 'random_state': RANDOM_STATE, 'verbose': 0}

# --- CANDIDATE SELECTION ---
CANDIDATES_A = [
    {"Rank": 1, "Feat": "Binary_ANOVA", "Model": "LightGBM",     "Resampler": "None"},
    {"Rank": 2, "Feat": "Binary_LGBM",  "Model": "RandomForest", "Resampler": "None"},
    {"Rank": 3, "Feat": "Binary_RFECV", "Model": "RandomForest", "Resampler": "None"},
    {"Rank": 4, "Feat": "Binary_RFECV", "Model": "XGBoost",      "Resampler": "None"},
    {"Rank": 5, "Feat": "Binary_LGBM",  "Model": "LightGBM",     "Resampler": "None"}
]

CANDIDATES_B = [
    {"Rank": 1, "Feat": "Binary_ANOVA", "Model": "LightGBM",     "Resampler": "SMOTE"},
    {"Rank": 2, "Feat": "Binary_ANOVA", "Model": "LightGBM",     "Resampler": "SMOTE_Tomek"},
    {"Rank": 3, "Feat": "Binary_LGBM",  "Model": "RandomForest", "Resampler": "SMOTE"},
    {"Rank": 4, "Feat": "Binary_LGBM",  "Model": "RandomForest", "Resampler": "SMOTE_Tomek"},
    {"Rank": 5, "Feat": "Binary_RFECV", "Model": "RandomForest", "Resampler": "SMOTE"}
]

CANDIDATES_C = [
    {"Rank": 1, "Feat": "FULL_FEATURES", "Model": "RandomForest",     "Resampler": "None"},
    {"Rank": 2, "Feat": "FULL_FEATURES", "Model": "XGBoost",          "Resampler": "None"},
    {"Rank": 3, "Feat": "FULL_FEATURES", "Model": "LightGBM",         "Resampler": "None"},
    {"Rank": 4, "Feat": "FULL_FEATURES", "Model": "GradientBoosting", "Resampler": "None"}
]

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_model_instance(name):
    if name == "RandomForest": return RandomForestClassifier(**rf_p)
    if name == "XGBoost": return xgb.XGBClassifier(**xgb_p)
    if name == "LightGBM": return lgb.LGBMClassifier(**lgb_p)
    if name == "GradientBoosting": return GradientBoostingClassifier(**gb_p)
    return None

def get_resampler_instance(name, k):
    if name == "None": return None
    base = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    if name == "SMOTE": return base
    if name == "SMOTE_Tomek": return SMOTETomek(random_state=RANDOM_STATE, smote=base)
    return None

def optimize_threshold_oof(probs, y_true):
    """Finds the threshold that maximizes MCC on Out-Of-Fold probabilities."""
    best_mcc = -1
    best_t = 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = t
    return best_t

def get_full_metrics(y_true, y_pred, threshold):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Binary MCC via pure formula
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom != 0 else 0.0
    
    # Calculate Per-Class MCC approximation (Rest = 0)
    denom0 = np.sqrt(float((tn+fn)*(tn+fp)*(tp+fn)*(tp+fp)))
    mcc0 = (tn*tp - fn*fp)/denom0 if denom0!=0 else 0.0
    
    return {
        "Global_MCC": mcc,
        "Rest_MCC": mcc0,
        "Tri_MCC": mcc,
        "Threshold": threshold
    }

# ==============================================================================
# 3. LOAD DATA & PREPARE FEATURES
# ==============================================================================
print(f"[*] Loading Data from {DATA_DIR}...")
train_df = pd.read_excel(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_excel(os.path.join(DATA_DIR, TEST_FILE))

y_train_full = train_df[TARGET_COLUMN].astype(int)
y_test_full  = test_df[TARGET_COLUMN].astype(int)
X_train_full = train_df.drop(columns=[TARGET_COLUMN])
X_test_full  = test_df.drop(columns=[TARGET_COLUMN])

# SMOTE k calc (Dynamic)
min_class_count = Counter(y_train_full).most_common()[-1][1]
safe_k = min(5, max(1, min_class_count - 1))
print(f" -> Data Loaded. Minority Count: {min_class_count}. SMOTE k_neighbors={safe_k}")

print("\n[*] Pre-loading Feature Sets...")
FEATURE_LISTS = {}

for feat_name, json_file in FEAT_FILES.items():
    f_path = os.path.join(DATA_DIR, json_file)
    try:
        with open(f_path, 'r') as f:
            FEATURE_LISTS[feat_name] = json.load(f)
    except Exception as e:
        print(f" [!] Warning: Could not load {json_file}. Did you run Step 10?")

# Explicitly assign full features for Branch C
FEATURE_LISTS["FULL_FEATURES"] = X_train_full.columns.tolist()

# ==============================================================================
# 4. OPTIMIZATION LOOP
# ==============================================================================
def run_optimization(candidates, branch_name):
    print(f"\n" + "="*95)
    print(f" STARTING {branch_name} (CV-Based Threshold Tuning)")
    print("="*95)
    
    results = []
    
    for cand in candidates:
        feat_name = cand['Feat']
        model_name = cand['Model']
        res_name = cand['Resampler']
        
        print(f" -> Optimizing Rank {cand['Rank']:<2}: {model_name:<16} | {feat_name:<15} | {res_name}")
        
        # 1. Load Features
        valid_feats = [f for f in FEATURE_LISTS[feat_name] if f in X_train_full.columns]
        X_tr_s = X_train_full[valid_feats]
        X_te_s = X_test_full[valid_feats]
        
        # 2. Build Pipeline
        steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]
        res_inst = get_resampler_instance(res_name, safe_k)
        if res_inst: steps.append(('resampler', res_inst))
        steps.append(('clf', get_model_instance(model_name)))
        
        pipeline = ImbPipeline(steps)
        
        # 3. Generate OOF Probabilities (Leakage Proof)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        try:
            oof_probs = cross_val_predict(pipeline, X_tr_s, y_train_full, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
        except Exception as e:
            print(f"      [!] Parallel execution failed. Falling back to single core (n_jobs=1)...")
            oof_probs = cross_val_predict(pipeline, X_tr_s, y_train_full, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
            
        # 4. Find Best Threshold
        best_t = optimize_threshold_oof(oof_probs, y_train_full)
        
        # 5. Final Fit on Full Train
        pipeline.fit(X_tr_s, y_train_full)
        
        # 6. Predict on Test
        probs_test = pipeline.predict_proba(X_te_s)[:, 1]
        
        # 7. Apply Optimized Threshold
        final_preds = (probs_test >= best_t).astype(int)
        
        # 8. Metrics
        metrics = get_full_metrics(y_test_full, final_preds, best_t)
        
        results.append({
            "Rank": cand['Rank'],
            "Feature": feat_name,
            "Model": model_name,
            "Resampler": res_name,
            "Opt_Threshold": best_t,
            "Global_MCC": metrics['Global_MCC'],
            "Rest_MCC": metrics['Rest_MCC'],
            "Tri_MCC": metrics['Tri_MCC']
        })
        
    return pd.DataFrame(results)

# Run All Branches
df_A = run_optimization(CANDIDATES_A, "BRANCH A (Reduced Features - No Resampling)")
df_B = run_optimization(CANDIDATES_B, "BRANCH B (Reduced Features - With Resampling)")
df_C = run_optimization(CANDIDATES_C, "BRANCH C (Full Features - No Resampling)")

# ==============================================================================
# 5. FINAL REPORT
# ==============================================================================
cols = ['Rank', 'Feature', 'Model', 'Resampler', 'Opt_Threshold', 'Global_MCC', 'Rest_MCC', 'Tri_MCC']

print("\n" + "="*100)
print(" FINAL RESULTS: BRANCH A ")
print("="*100)
print(df_A[cols].to_string(index=False, float_format="%.4f"))

print("\n" + "="*100)
print(" FINAL RESULTS: BRANCH B ")
print("="*100)
print(df_B[cols].to_string(index=False, float_format="%.4f"))

print("\n" + "="*100)
print(" FINAL RESULTS: BRANCH C ")
print("="*100)
print(df_C[cols].to_string(index=False, float_format="%.4f"))

# Save Outputs
df_A.to_csv(os.path.join(BASE_DIR, "13_Binary_Final_Opt_Branch_A.csv"), index=False)
df_B.to_csv(os.path.join(BASE_DIR, "13_Binary_Final_Opt_Branch_B.csv"), index=False)
df_C.to_csv(os.path.join(BASE_DIR, "13_Binary_Final_Opt_Branch_C.csv"), index=False)

print("\n✅ Binary Threshold Optimization Complete. All results successfully exported to CSV.")