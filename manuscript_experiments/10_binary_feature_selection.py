"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 10: Binary Feature Selection Protocol (Trimer Specific)
==============================================================================
Description:
Executes 3 distinct feature selection methodologies to prune the RBF feature space 
for the binary classification task (Target: State_tri):
1. ANOVA (SelectKBest)
2. LightGBM Importance (Threshold='median')
3. RFECV (Recursive Feature Elimination with CV)

Methodology:
- Protocol: Impute -> Scale -> SMOTE -> Select -> Evaluate
- Validation: 3-Fold Stratified CV (Leakage-Proof via Imblearn Pipeline)
- Dynamic SMOTE: k_neighbors dynamically adjusted based on minority class size.
- Output: 3 JSON files containing the definitive feature subsets.
==============================================================================
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFECV
from sklearn.metrics import matthews_corrcoef, make_scorer

# Models
import lightgbm as lgb

# Resampling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================
# Dynamically locate the DATASETS/binary folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS", "binary")

TRAIN_FILE = "BINARY_RBF_TRAIN.xlsx"
TEST_FILE  = "BINARY_RBF_TEST.xlsx"
TARGET_COLUMN = "State_tri" 
RANDOM_STATE = 42

# Optimized LGBM Params (Full Tuned)
lgb_params = {
    'n_estimators': 860,
    'learning_rate': 0.0464,
    'num_leaves': 30,
    'max_depth': 13,
    'subsample': 0.806,
    'colsample_bytree': 0.673,
    'reg_alpha': 1e-08,
    'reg_lambda': 0.172,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': -1
}

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
print(f"[*] Loading Binary Data from {DATA_DIR}...")
train_path = os.path.join(DATA_DIR, TRAIN_FILE)

if not os.path.exists(train_path):
    print(f"[!] ERROR: {TRAIN_FILE} not found. Please check your DATASETS/binary folder.")
    exit()

train_df = pd.read_excel(train_path)

y_train = train_df[TARGET_COLUMN].astype(int)
X_train = train_df.drop(columns=[TARGET_COLUMN])

# SMOTE Neighbor Calculation
min_class_size = Counter(y_train).most_common()[-1][1]
k_neigh = min(5, max(1, min_class_size - 1))
print(f" -> Data Loaded. Minority Class Size: {min_class_size}. SMOTE safe k_neighbors={k_neigh}\n")

# ==============================================================================
# 3. DEFINE SELECTORS
# ==============================================================================
# Fast LGBM for RFECV recursion (250 trees for optimal quality/speed balance)
lgb_fast = lgb.LGBMClassifier(**lgb_params)
lgb_fast.set_params(n_estimators=250) 

# Explicit Stratified CV 
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

selectors = {
    "Binary_ANOVA": SelectKBest(score_func=f_classif, k=X_train.shape[1]//2),
    
    "Binary_LGBM_Importance": SelectFromModel(
        lgb.LGBMClassifier(**lgb_params), # Uses FULL 860 trees
        threshold='median'
    ),
    
    "Binary_RFECV": RFECV(
        estimator=lgb_fast,  
        step=5,              
        cv=cv_strategy,      
        scoring='matthews_corrcoef', 
        n_jobs=-1
    )
}

# ==============================================================================
# 4. EXECUTION LOOP
# ==============================================================================
print("="*80)
print(" STARTING BINARY FEATURE SELECTION PROTOCOL ")
print("="*80)

for name, selector in selectors.items():
    start_ts = time.time()
    print(f"\n--- Processing: {name} ---")
    
    # -------------------------------------------------------
    # A. Robust Benchmarking (Leakage-Proof CV)
    # -------------------------------------------------------
    benchmark_pipe = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neigh)),
        ('selector', selector),
        ('clf', lgb.LGBMClassifier(**lgb_params)) # Evaluates using FULL model
    ])
    
    print("   Evaluating subset via 3-Fold Stratified CV...")
    cv_scores = cross_val_score(benchmark_pipe, X_train, y_train, cv=cv_strategy, scoring='matthews_corrcoef', n_jobs=-1)
    avg_mcc = np.mean(cv_scores)
    print(f"   Average CV MCC: {avg_mcc:.4f}")
    
    # -------------------------------------------------------
    # B. Extraction & Saving
    # -------------------------------------------------------
    print(f"   Extracting final feature list...")
    
    # Pipeline for extraction
    extraction_pipe = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neigh)),
        ('selector', selector)
    ])
    
    # Fit on full X_train to get the definitive list
    extraction_pipe.fit(X_train, y_train)
    
    mask = extraction_pipe.named_steps['selector'].get_support()
    selected_feats = X_train.columns[mask].tolist()
    
    # Save directly to the binary DATASETS folder
    filename = f"{name}_Features.json"
    save_path = os.path.join(DATA_DIR, filename)
    
    with open(save_path, "w") as f:
        json.dump(selected_feats, f)
        
    duration = time.time() - start_ts
    print(f"   ✅ Retained {len(selected_feats)} out of {X_train.shape[1]} features.")
    print(f"   ✅ Saved feature list to: {filename}")
    print(f"   Time taken: {duration:.1f}s")

# ==============================================================================
# 5. COMPLETION
# ==============================================================================
print("\n" + "="*80)
print(" BINARY FEATURE SELECTION PIPELINE COMPLETE ")
print("="*80)