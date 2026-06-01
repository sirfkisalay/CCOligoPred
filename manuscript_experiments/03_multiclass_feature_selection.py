"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 3: Multiclass Feature Selection Protocol
==============================================================================
Description:
Executes 6 distinct feature selection methodologies to prune the RBF feature space:
1. ANOVA (SelectKBest)
2. RFECV (Recursive Feature Elimination with CV)
3. LightGBM Importance (Threshold='mean')
4. XGBoost Importance (Threshold='mean')
5. Gradient Boosting Importance (Threshold='mean')
6. ExtraTrees Importance (Threshold='mean')

Methodology:
- Protocol: Impute -> Scale -> Select -> Evaluate
- Validation: 3-Fold Stratified CV (Leakage-Proof)
- Bias Mitigation: Class weighting used over SMOTE to prevent synthetic bias.
- Output: 6 JSON files containing the definitive feature subsets.
==============================================================================
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFECV
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.pipeline import Pipeline 

# Models
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS")
TRAIN_FILE_PATH = os.path.join(DATA_DIR, "MULTICLASS_RBF_TRAIN.xlsx")

TARGET_COLUMN = "Class" 
RANDOM_STATE = 42

mcc_scorer = make_scorer(matthews_corrcoef)

# --- OPTIMIZED HYPERPARAMETERS (From Step 2) ---
lgb_params = {
    'n_estimators': 1250, 
    'max_depth': 23, 
    'learning_rate': 0.04472528989077509, 
    'num_leaves': 113, 
    'subsample': 0.9786298784381904, 
    'colsample_bytree': 0.8115594723131374, 
    'min_child_samples': 20, 
    'class_weight': 'balanced',
    'objective': 'multiclass',
    'num_class': 4,
    'n_jobs': -1,
    'verbosity': -1,
    'random_state': RANDOM_STATE
}

xgb_params = {
    'n_estimators': 700, 
    'max_depth': 11, 
    'learning_rate': 0.0147951583299575, 
    'subsample': 0.7776832317133678, 
    'colsample_bytree': 0.5870074173438061, 
    'gamma': 0.9200227958551773, 
    'min_child_weight': 4,
    'objective': 'multi:softmax',
    'num_class': 4,
    'eval_metric': 'mlogloss',
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}

gb_params = {
    'n_estimators': 1900, 
    'learning_rate': 0.18698360549487777, 
    'max_depth': 9, 
    'min_samples_split': 20, 
    'min_samples_leaf': 15, 
    'subsample': 0.8386502590161441, 
    'max_features': 'log2',
    'random_state': RANDOM_STATE
}

et_params = {
    'n_estimators': 600, 
    'max_depth': 50, 
    'min_samples_split': 15, 
    'min_samples_leaf': 2, 
    'max_features': None, 
    'bootstrap': True, 
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}

# Fast LGBM for RFECV recursion (capped to save time)
lgb_fast = lgb.LGBMClassifier(**lgb_params)
lgb_fast.set_params(n_estimators=250) 

# ==============================================================================
# 2. LOAD & PREPARE DATA
# ==============================================================================
print(f"[*] Loading Multiclass Training Data...")
if not os.path.exists(TRAIN_FILE_PATH):
    print(f"[!] ERROR: Data file not found at {TRAIN_FILE_PATH}")
    exit()

train_df = pd.read_excel(TRAIN_FILE_PATH)
y_train_raw = train_df[TARGET_COLUMN]
X_train = train_df.drop(columns=[TARGET_COLUMN])

# --- MULTICLASS TARGET MAPPING & ENCODING ---
class_name_map = {1: 'PD', 2: 'APD', 3: 'TRI', 4: 'TET'}
y_train_mapped = y_train_raw.map(class_name_map).fillna(y_train_raw)

encoder = LabelEncoder()
y_train = pd.Series(encoder.fit_transform(y_train_mapped))
print(f" -> Class Mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
print(f" -> Initial Feature Shape: {X_train.shape}\n")

# ==============================================================================
# 3. DEFINE SELECTOR PROTOCOLS
# ==============================================================================
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

protocols = {
    "Multiclass_ANOVA": {
        "selector": SelectKBest(score_func=f_classif, k=X_train.shape[1]//2),
        "evaluator": lgb.LGBMClassifier(**lgb_params)
    },
    "Multiclass_RFECV": {
        "selector": RFECV(estimator=lgb_fast, step=5, cv=cv_strategy, scoring=mcc_scorer, n_jobs=-1),
        "evaluator": lgb.LGBMClassifier(**lgb_params)
    },
    "Multiclass_LGBM_Importance": {
        "selector": SelectFromModel(lgb.LGBMClassifier(**lgb_params), threshold='mean'),
        "evaluator": lgb.LGBMClassifier(**lgb_params)
    },
    "Multiclass_XGB_Importance": {
        "selector": SelectFromModel(xgb.XGBClassifier(**xgb_params), threshold='mean'),
        "evaluator": xgb.XGBClassifier(**xgb_params)
    },
    "Multiclass_GB_Importance": {
        "selector": SelectFromModel(GradientBoostingClassifier(**gb_params), threshold='mean'),
        "evaluator": GradientBoostingClassifier(**gb_params)
    },
    "Multiclass_ExtraTrees_Importance": {
        "selector": SelectFromModel(ExtraTreesClassifier(**et_params), threshold='mean'),
        "evaluator": ExtraTreesClassifier(**et_params)
    }
}

# ==============================================================================
# 4. EXECUTION LOOP
# ==============================================================================
print("="*60)
print(" STARTING FEATURE SELECTION PROTOCOLS ")
print("="*60)

for name, components in protocols.items():
    start_ts = time.time()
    print(f"\n--- Processing: {name} ---")
    
    current_selector = components["selector"]
    current_evaluator = components["evaluator"]
    
    # -------------------------------------------------------
    # A. Robust Benchmarking (Leakage-Proof CV)
    # -------------------------------------------------------
    benchmark_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()),
        ('selector', current_selector),
        ('clf', current_evaluator)
    ])
    
    print("   Evaluating subset via 3-Fold Stratified CV...")
    cv_scores = cross_val_score(benchmark_pipe, X_train, y_train, cv=cv_strategy, scoring=mcc_scorer, n_jobs=-1)
    avg_mcc = np.mean(cv_scores)
    print(f"   Average CV Multiclass MCC: {avg_mcc:.4f}")
    
    # -------------------------------------------------------
    # B. Extraction & Saving
    # -------------------------------------------------------
    print(f"   Extracting final feature list...")
    extraction_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', current_selector)
    ])
    
    # Fit on full X_train to lock in the definitive list
    extraction_pipe.fit(X_train, y_train)
    
    mask = extraction_pipe.named_steps['selector'].get_support()
    selected_feats = X_train.columns[mask].tolist()
    
    # Save the selected features to a JSON file
    filename = f"{name}_Features_Raw.json"
    save_path = os.path.join(BASE_DIR, filename)
    
    with open(save_path, "w") as f:
        json.dump(selected_feats, f)
        
    duration = time.time() - start_ts
    print(f"   ✅ Retained {len(selected_feats)} out of {X_train.shape[1]} features.")
    print(f"   ✅ Saved to: {filename}")
    print(f"   Time taken: {duration:.1f}s")

# ==============================================================================
# 5. COMPLETION
# ==============================================================================
print("\n" + "="*60)
print(" MULTICLASS FEATURE SELECTION PIPELINE COMPLETE ")
print("="*60)