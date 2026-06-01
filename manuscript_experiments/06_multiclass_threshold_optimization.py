"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 6: Multiclass Final Threshold Optimization
==============================================================================
Description:
Performs Out-Of-Fold (OOF) cross-validation to derive optimal decision 
thresholds for each specific oligomeric state (PD, APD, TRI, TET). 
This counteracts the natural bias of classifiers toward the majority classes.

Evaluates three pre-selected experimental branches:
- BRANCH A: Top 5 Models (Reduced Features, No Resampling)
- BRANCH B: Top 5 Models (Reduced Features, With Resampling)
- BRANCH C: Baseline (Full Features, No Resampling)

Methodology:
- Probabilities generated via 5-Fold Stratified CV (Leakage-Proof).
- Thresholds optimized dynamically (0.05 to 0.95) to maximize per-class MCC.
- Final test predictions scaled via threshold margin division.
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# Models
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

# Resampling
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Dynamically locate the DATASETS folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS")

TRAIN_FILE = "MULTICLASS_RBF_TRAIN.xlsx"
TEST_FILE  = "MULTICLASS_RBF_TEST.xlsx"
TARGET_COLUMN = "Class"
CLASS_NAMES = ['PD', 'APD', 'TRI', 'TET']
RANDOM_STATE = 42
CV_FOLDS = 5 

# Feature Files Map
FEAT_FILES = {
    "Multi_ANOVA": "Multiclass_ANOVA_Features_Raw.json",
    "Multi_RFECV": "Multiclass_RFECV_Features_Raw.json",
    "Multi_GB": "Multiclass_GB_Importance_Features_mean_Raw.json",
    "Multi_XGB": "Multiclass_XGB_Importance_Features_mean_Raw.json",
    "Multi_LGBM": "Multiclass_LGBM_Importance_Features_mean_Raw.json",
    "Multi_ExtraTrees": "Multiclass_ExtraTrees_Importance_Features_Raw.json"
}

# --- OPTIMIZED HYPERPARAMETERS ---
xgb_p = {
    'n_estimators': 700, 'max_depth': 11, 'learning_rate': 0.0147951583299575, 
    'subsample': 0.7776832317133678, 'colsample_bytree': 0.5870074173438061, 
    'gamma': 0.9200227958551773, 'min_child_weight': 4,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0,
    'objective': 'multi:softprob', 'eval_metric': 'mlogloss' 
}

lgb_p = {
    'n_estimators': 1250, 'max_depth': 23, 'learning_rate': 0.04472528989077509, 
    'num_leaves': 113, 'subsample': 0.9786298784381904, 'colsample_bytree': 0.8115594723131374, 
    'min_child_samples': 20, 'class_weight': 'balanced',
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1,
    'objective': 'multiclass' 
}

gb_p = {
    'n_estimators': 1900, 'learning_rate': 0.18698360549487777, 'max_depth': 9, 
    'min_samples_split': 20, 'min_samples_leaf': 15, 'subsample': 0.8386502590161441, 
    'max_features': 'log2',
    'random_state': RANDOM_STATE
}

et_p = {
    'n_estimators': 600, 'max_depth': 50, 'min_samples_split': 15, 
    'min_samples_leaf': 2, 'max_features': None, 'bootstrap': True, 
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': 0
}

# --- CANDIDATE SELECTION (TOP 5 ONLY) ---

CANDIDATES_A = [
    {"Rank": 1,  "Feat": "Multi_XGB",        "Model": "ExtraTrees",       "Resampler": "None"},
    {"Rank": 2,  "Feat": "Multi_RFECV",      "Model": "ExtraTrees",       "Resampler": "None"},
    {"Rank": 3,  "Feat": "Multi_GB",         "Model": "ExtraTrees",       "Resampler": "None"},
    {"Rank": 4,  "Feat": "Multi_LGBM",       "Model": "ExtraTrees",       "Resampler": "None"},
    {"Rank": 5,  "Feat": "Multi_XGB",        "Model": "GradientBoosting", "Resampler": "None"}
]

CANDIDATES_B = [
    {"Rank": 1,  "Feat": "Multi_ExtraTrees", "Model": "GradientBoosting", "Resampler": "SMOTE_Tomek"},
    {"Rank": 2,  "Feat": "Multi_XGB",        "Model": "LightGBM",         "Resampler": "SMOTE_Tomek"},
    {"Rank": 3,  "Feat": "Multi_GB",         "Model": "XGBoost",          "Resampler": "SMOTE_Tomek"},
    {"Rank": 4,  "Feat": "Multi_ExtraTrees", "Model": "LightGBM",         "Resampler": "SMOTE"},
    {"Rank": 5,  "Feat": "Multi_XGB",        "Model": "LightGBM",         "Resampler": "SMOTE"}
]

CANDIDATES_C = [
    {"Rank": 1, "Feat": "FULL_FEATURES", "Model": "XGBoost",          "Resampler": "None"},
    {"Rank": 2, "Feat": "FULL_FEATURES", "Model": "LightGBM",         "Resampler": "None"},
    {"Rank": 3, "Feat": "FULL_FEATURES", "Model": "GradientBoosting", "Resampler": "None"},
    {"Rank": 4, "Feat": "FULL_FEATURES", "Model": "ExtraTrees",       "Resampler": "None"}
]

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_model_instance(name):
    if name == "XGBoost": return xgb.XGBClassifier(**xgb_p)
    if name == "LightGBM": return lgb.LGBMClassifier(**lgb_p)
    if name == "GradientBoosting": return GradientBoostingClassifier(**gb_p)
    if name == "ExtraTrees": return ExtraTreesClassifier(**et_p)
    return None

def get_resampler_instance(name, k):
    if name == "None": return None
    base = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    if name == "SMOTE": return base
    if name == "SMOTE_Tomek": return SMOTETomek(random_state=RANDOM_STATE, smote=base)
    return None

def optimize_threshold_multiclass_oof(probs, y_true, num_classes):
    best_thresholds = np.full(num_classes, 0.5)
    for c in range(num_classes):
        c_probs = probs[:, c]
        c_y_true = (y_true == c).astype(int)
        
        best_mcc = -1
        best_t = 0.5
        
        for t in np.arange(0.05, 0.96, 0.05): 
            c_preds = (c_probs >= t).astype(int)
            mcc = matthews_corrcoef(c_y_true, c_preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_t = t
                
        best_thresholds[c] = best_t
    return best_thresholds

def apply_multiclass_thresholds(probs, thresholds):
    """ Scales probabilities using the optimized thresholds before argmax """
    scaled_probs = probs / thresholds
    return np.argmax(scaled_probs, axis=1)

def get_full_metrics(y_true, y_pred, classes, thresholds):
    """ Generates Global MCC, MOSS Score, Per-Class MCC, and Thresholds """
    cm = confusion_matrix(y_true, y_pred)
    global_mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics = {
        "Global_MCC": global_mcc,
        "Opt_Thresholds": str(np.round(thresholds, 2).tolist())
    }
    
    pc_mcc = {}
    for i, cls_name in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = (tp * tn - fp * fn) / denom if denom != 0 else 0.0
        
        pc_mcc[cls_name] = mcc
        metrics[f"MCC_{cls_name}"] = mcc
        
    # Calculate MOSS Score
    moss_score = (global_mcc + pc_mcc.get("TRI", 0.0) + pc_mcc.get("TET", 0.0)) / 3
    metrics["MOSS_Score"] = moss_score
        
    return metrics

# ==============================================================================
# 3. LOAD DATA & PREPARE FEATURE DICTIONARY
# ==============================================================================
print(f"[*] Loading Data from {DATA_DIR}...")
train_df = pd.read_excel(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_excel(os.path.join(DATA_DIR, TEST_FILE))

y_train_raw = train_df[TARGET_COLUMN]
y_test_raw  = test_df[TARGET_COLUMN]

# Map targets consistently
class_name_map = {1: 'PD', 2: 'APD', 3: 'TRI', 4: 'TET'}
y_train_mapped = y_train_raw.map(class_name_map).fillna(y_train_raw)
y_test_mapped = y_test_raw.map(class_name_map).fillna(y_test_raw)

# Encode
le = LabelEncoder()
y_train_full = le.fit_transform(y_train_mapped)
y_test_full = le.transform(y_test_mapped)
num_classes = len(np.unique(y_train_full))

X_train_full = train_df.drop(columns=[TARGET_COLUMN])
X_test_full  = test_df.drop(columns=[TARGET_COLUMN])

min_class_count = Counter(y_train_full).most_common()[-1][1]
safe_k = min(5, max(1, min_class_count - 1))
print(f" -> Classes: {num_classes} ({le.classes_}). SMOTE safe k_neighbors={safe_k}")

print("\n[*] Pre-loading Feature Sets...")
FEATURE_LISTS = {}

# Load all JSON reduced sets
for feat_name, json_file in FEAT_FILES.items():
    f_path = os.path.join(DATA_DIR, json_file)
    try:
        with open(f_path, 'r') as f:
            FEATURE_LISTS[feat_name] = json.load(f)
    except Exception as e:
        print(f" [!] Warning: Could not load {json_file}. Did you run Step 3?")

# EXPLICITLY DEFINE FULL FEATURES HERE
FEATURE_LISTS["FULL_FEATURES"] = X_train_full.columns.tolist()

# ==============================================================================
# 4. OPTIMIZATION LOOP
# ==============================================================================
def run_optimization(candidates, branch_name):
    print(f"\n" + "="*110)
    print(f" STARTING {branch_name} ")
    print("="*110)
    
    results = []
    
    for cand in candidates:
        feat_name = cand['Feat']
        model_name = cand['Model']
        res_name = cand['Resampler']
        
        print(f" -> Optimizing Rank {cand['Rank']:<2}: {model_name:<16} | {feat_name:<16} | {res_name}")
        
        # 1. Get Features Directly From Dictionary
        valid_feats = [f for f in FEATURE_LISTS[feat_name] if f in X_train_full.columns]
        X_tr_s = X_train_full[valid_feats]
        X_te_s = X_test_full[valid_feats]
        
        # 2. Build Pipeline
        steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]
        res_inst = get_resampler_instance(res_name, safe_k)
        if res_inst: steps.append(('resampler', res_inst))
        steps.append(('clf', get_model_instance(model_name)))
        
        pipeline = ImbPipeline(steps)
        
        # 3. Generate OOF Probabilities
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        try:
            oof_probs = cross_val_predict(pipeline, X_tr_s, y_train_full, cv=cv, method='predict_proba', n_jobs=-1)
        except:
            oof_probs = cross_val_predict(pipeline, X_tr_s, y_train_full, cv=cv, method='predict_proba', n_jobs=1)
            
        # 4. Find Best Threshold Array
        best_thresholds = optimize_threshold_multiclass_oof(oof_probs, y_train_full, num_classes)
        
        # 5. Final Fit on Full Train
        pipeline.fit(X_tr_s, y_train_full)
        
        # 6. Predict Probabilities on Test
        probs_test = pipeline.predict_proba(X_te_s)
        
        # 7. Apply Optimized Thresholds via Margin Scaling
        final_preds = apply_multiclass_thresholds(probs_test, best_thresholds)
        
        # 8. Metrics
        metrics = get_full_metrics(y_test_full, final_preds, CLASS_NAMES, best_thresholds)
        
        results.append({
            "Rank": cand['Rank'],
            "Feature": feat_name,
            "Model": model_name,
            "Resampler": res_name,
            **metrics
        })
        
    return pd.DataFrame(results)

# Run All Branches
df_A = run_optimization(CANDIDATES_A, "BRANCH A (Top 5 - No Resampling)")
df_B = run_optimization(CANDIDATES_B, "BRANCH B (Top 5 - With Resampling)")
df_C = run_optimization(CANDIDATES_C, "BRANCH C (Baseline - Full Features)")

# ==============================================================================
# 5. FINAL REPORT
# ==============================================================================
# Reordering columns for a cleaner CSV output
col_order = ['Rank', 'Feature', 'Model', 'Resampler', 'MOSS_Score', 'Global_MCC', 
             'MCC_PD', 'MCC_APD', 'MCC_TRI', 'MCC_TET', 'Opt_Thresholds']

df_A = df_A[col_order]
df_B = df_B[col_order]
df_C = df_C[col_order]

print("\n" + "="*120)
print(" FINAL RESULTS: BRANCH A ")
print("="*120)
print(df_A.to_string(index=False))

print("\n" + "="*120)
print(" FINAL RESULTS: BRANCH B ")
print("="*120)
print(df_B.to_string(index=False))

print("\n" + "="*120)
print(" FINAL RESULTS: BRANCH C ")
print("="*120)
print(df_C.to_string(index=False))

# Save Outputs
df_A.to_csv(os.path.join(BASE_DIR, "06_Multiclass_Final_Opt_Branch_A.csv"), index=False)
df_B.to_csv(os.path.join(BASE_DIR, "06_Multiclass_Final_Opt_Branch_B.csv"), index=False)
df_C.to_csv(os.path.join(BASE_DIR, "06_Multiclass_Final_Opt_Branch_C.csv"), index=False)

print("\n✅ Threshold Optimization Complete. All Branch results successfully exported to CSV.")