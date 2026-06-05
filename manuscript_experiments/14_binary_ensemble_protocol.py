"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 14: Binary Ensemble Protocol (Bagging, Voting, Stacking)
==============================================================================
Description:
Evaluates advanced ensemble meta-learning strategies using the full Register-
Based Features (RBF) dataset for the binary classification task (Target: State_tri).
Combines the predictive power of the top optimized baseline architectures 
(LightGBM, XGBoost, and RandomForest).

Methodologies Tested:
1. Bagging: Reduces variance of the champion LightGBM model.
2. Soft Voting: Weighted probability consensus (LGBM=2, XGB=1, RF=1).
3. Stacking: Meta-learning via Logistic Regression over OOF predictions.

Outputs: '14_Binary_Ensemble_Results.csv'
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import warnings

# Metrics
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Ensembles
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Dynamically locate the DATASETS/binary folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS", "binary")

TRAIN_FILE = "BINARY_RBF_TRAIN.xlsx"
TEST_FILE  = "BINARY_RBF_TEST.xlsx"
TARGET_COLUMN = "State_tri"
CLASS_NAMES = ['Rest', 'Tri']
RANDOM_STATE = 42

# --- OPTIMIZED HYPERPARAMETERS (From previous tuning steps) ---
rf_p = {
    'n_estimators': 1300, 'max_depth': 23, 'min_samples_leaf': 9, 'max_features': 0.5, 
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': 0
}

xgb_p = {
    'n_estimators': 1150, 'max_depth': 12, 'learning_rate': 0.0405246, 
    'subsample': 0.75433, 'colsample_bytree': 0.60561, 
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0, 
    'objective': 'binary:logistic', 'eval_metric': 'logloss'
}

lgb_p = {
    'n_estimators': 860, 'learning_rate': 0.046423, 'num_leaves': 30, 'max_depth': 13, 
    'subsample': 0.80635, 'colsample_bytree': 0.67299, 
    'reg_alpha': 1.1078e-08, 'reg_lambda': 0.17206, 
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1, 
    'objective': 'binary'
}

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def calculate_per_class_mcc(y_true, y_pred, classes):
    """Calculates MCC for each class (Symmetric in binary)."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Global/Tri MCC
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc_tri = (tp * tn - fp * fn) / denom if denom != 0 else 0.0
    
    # Rest MCC (Symmetric in binary logic)
    mcc_rest = mcc_tri
    
    return mcc_tri, mcc_rest

# ==============================================================================
# 3. DATA LOADING (FULL FEATURES)
# ==============================================================================
print(f"[*] Loading Full Datasets from {DATA_DIR}...")
train_path = os.path.join(DATA_DIR, TRAIN_FILE)
test_path = os.path.join(DATA_DIR, TEST_FILE)

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print(f"[!] ERROR: Datasets not found. Please check your DATASETS/binary folder.")
    exit()

train_df = pd.read_excel(train_path)
test_df  = pd.read_excel(test_path)

y_train = train_df[TARGET_COLUMN].astype(int)
y_test  = test_df[TARGET_COLUMN].astype(int)
X_train = train_df.drop(columns=[TARGET_COLUMN])
X_test  = test_df.drop(columns=[TARGET_COLUMN])

X_test = X_test[X_train.columns] # Align columns

print(f" -> Data Loaded. Full Features Used: {X_train.shape[1]}\n")

# ==============================================================================
# 4. BUILDING INDIVIDUAL PIPELINES
# ==============================================================================
print("[*] Constructing Base Pipelines...")

# Wrapping each model in a pipeline ensures that during Stacking/Bagging, 
# imputation and scaling happen INSIDE the CV folds, preventing leakage.
pipe_lgb = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', lgb.LGBMClassifier(**lgb_p))
])

pipe_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', xgb.XGBClassifier(**xgb_p))
])

pipe_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(**rf_p))
])

# ==============================================================================
# 5. DEFINING ENSEMBLES
# ==============================================================================
ensembles = {}

# A. Bagging (Primary Model: LightGBM)
# ------------------------------------
ensembles["Bagging_LGBM"] = BaggingClassifier(
    estimator=pipe_lgb,
    n_estimators=10,
    max_samples=0.8, # Train on 80% of data per estimator
    max_features=1.0,
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=1 # Inner jobs are already parallelized (-1)
)

# B. Soft Voting (Weighted)
# ------------------------------------
# 2 parts LightGBM, 1 part XGB, 1 part RF. (Favor the Champion)
ensembles["Voting_Weighted"] = VotingClassifier(
    estimators=[
        ('lgb', pipe_lgb),
        ('xgb', pipe_xgb),
        ('rf',  pipe_rf)
    ],
    voting='soft',
    weights=[2, 1, 1] 
)

# C. Stacking (Meta-Learning)
# ------------------------------------
# 5-Fold internal CV ensures leakage-proof training of the meta-learner.
ensembles["Stacking_CV"] = StackingClassifier(
    estimators=[
        ('lgb', pipe_lgb),
        ('xgb', pipe_xgb),
        ('rf',  pipe_rf)
    ],
    final_estimator=LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE),
    cv=5, 
    n_jobs=1,
    passthrough=False 
)

# ==============================================================================
# 6. TRAINING & EVALUATION LOOP
# ==============================================================================
print("[*] Training Ensembles (This may take a few minutes)...")
results_log = []

for name, clf in ensembles.items():
    print(f"\n   >>> Training Ensemble Engine: {name} ...")
    
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    preds = clf.predict(X_test)
    
    # Evaluate
    g_mcc = matthews_corrcoef(y_test, preds)
    mcc_tri, mcc_rest = calculate_per_class_mcc(y_test, preds, CLASS_NAMES)
    
    print(f"       Global MCC: {g_mcc:.4f}")
    
    results_log.append({
        "Ensemble_Type": name,
        "Global_MCC": g_mcc,
        "Rest_MCC": mcc_rest,
        "Tri_MCC": mcc_tri
    })

# ==============================================================================
# 7. FINAL COMPARISON
# ==============================================================================
print("\n" + "="*80)
print("🏆 ENSEMBLE RESULTS COMPARISON (Binary Task) 🏆")
print("="*80)

# Create DataFrame
df_res = pd.DataFrame(results_log).sort_values(by="Global_MCC", ascending=False)

# Add baseline for reference based on previous outputs
print(f"Reference Baseline (Single LightGBM Model): ~0.4592\n")

print(df_res.to_string(index=False, float_format="%.4f"))

# Save Outputs
out_file = os.path.join(BASE_DIR, "14_Binary_Ensemble_Results.csv")
df_res.to_csv(out_file, index=False)
print(f"\n✅ Results successfully generated and exported to:\n{out_file}")