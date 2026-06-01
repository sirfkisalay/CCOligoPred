"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 7: Multiclass Ensemble Protocol (Bagging, Voting, Stacking)
==============================================================================
Description:
Evaluates advanced ensemble meta-learning strategies using the full Register-
Based Features (RBF) dataset. Combines the predictive power of the top optimized 
baseline architectures (ExtraTrees, XGBoost, and LightGBM) to counteract the 
high variance typically observed in minority class predictions (TRI, TET).

Methodologies Tested:
1. Bagging: Reduces variance of the champion ExtraTrees model.
2. Soft Voting: Weighted probability consensus (ET=2, XGB=2, LGB=1).
3. Stacking: Meta-learning via Logistic Regression over OOF predictions.

Outputs: '07_Multiclass_Ensemble_Results.csv'
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import warnings

# Metrics
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Ensembles
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb

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
RANDOM_STATE = 42

# --- OPTIMIZED HYPERPARAMETERS (From Multiclass Tuning Steps) ---
et_p = {
    'n_estimators': 600, 'max_depth': 50, 'min_samples_split': 15, 
    'min_samples_leaf': 2, 'max_features': None, 'bootstrap': True, 
    'class_weight': 'balanced', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': 0
}

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

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def calculate_per_class_mcc(y_true, y_pred, classes):
    """Calculates MCC for each class using a One-vs-Rest approach."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_mcc = {}
    
    for i, cls_name in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = (tp * tn - fp * fn) / denom if denom != 0 else 0.0
        per_class_mcc[cls_name] = mcc
        
    return per_class_mcc

# ==============================================================================
# 3. DATA LOADING & ENCODING (FULL FEATURES)
# ==============================================================================
print(f"[*] Loading Full Datasets from {DATA_DIR}...")
try:
    train_df = pd.read_excel(os.path.join(DATA_DIR, TRAIN_FILE))
    test_df  = pd.read_excel(os.path.join(DATA_DIR, TEST_FILE))
except FileNotFoundError:
    print(f"[!] ERROR: {TRAIN_FILE} not found. Please check your DATASETS folder.")
    exit()

y_train_raw = train_df[TARGET_COLUMN]
y_test_raw  = test_df[TARGET_COLUMN]

# Map targets to their biological names
class_name_map = {1: 'PD', 2: 'APD', 3: 'TRI', 4: 'TET'}
y_train_mapped = y_train_raw.map(class_name_map).fillna(y_train_raw)
y_test_mapped = y_test_raw.map(class_name_map).fillna(y_test_raw)

# Label Encode Targets cleanly for multiclass compliance
le = LabelEncoder()
y_train = le.fit_transform(y_train_mapped)
y_test = le.transform(y_test_mapped)
CLASS_NAMES = le.classes_.tolist() # Will automatically sort to ['APD', 'PD', 'TET', 'TRI']

X_train = train_df.drop(columns=[TARGET_COLUMN])
X_test  = test_df.drop(columns=[TARGET_COLUMN])
X_test = X_test[X_train.columns]  # Align columns perfectly

print(f" -> Data Loaded. Features: {X_train.shape[1]} | Target Classes: {CLASS_NAMES}\n")

# ==============================================================================
# 4. BUILDING INDIVIDUAL PIPELINES
# ==============================================================================
print("[*] Constructing Leakage-Proof Base Pipelines...")

pipe_et = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', ExtraTreesClassifier(**et_p))
])

pipe_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', xgb.XGBClassifier(**xgb_p))
])

pipe_lgb = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', lgb.LGBMClassifier(**lgb_p))
])

# ==============================================================================
# 5. DEFINING ENSEMBLES
# ==============================================================================
ensembles = {}

# A. Bagging (Primary Model: ExtraTrees)
# ------------------------------------
ensembles["Bagging_ExtraTrees"] = BaggingClassifier(
    estimator=pipe_et,
    n_estimators=10,
    max_samples=0.8,  # Train on 80% data subsets to combat minority variance
    max_features=1.0,
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=1  # Pipeline inner estimators utilize internal n_jobs=-1
)

# B. Soft Voting (Weighted Consensus)
# ------------------------------------
# ExtraTrees (Champion) and XGBoost get higher weights, LightGBM stabilizes TRI class
ensembles["Voting_Weighted"] = VotingClassifier(
    estimators=[
        ('et',  pipe_et),
        ('xgb', pipe_xgb),
        ('lgb', pipe_lgb)
    ],
    voting='soft',
    weights=[2, 2, 1]
)

# C. Stacking (Meta-Learning)
# ------------------------------------
ensembles["Stacking_CV"] = StackingClassifier(
    estimators=[
        ('et',  pipe_et),
        ('xgb', pipe_xgb),
        ('lgb', pipe_lgb)
    ],
    final_estimator=LogisticRegression(multi_class='multinomial', class_weight='balanced', random_state=RANDOM_STATE),
    cv=5,  # Leakage-proof internal 5-fold cross validation
    n_jobs=1,
    passthrough=False 
)

# ==============================================================================
# 6. TRAINING & EVALUATION LOOP
# ==============================================================================
print("[*] Training Multiclass Ensembles (This may take a few minutes)...")
results_log = []

for name, clf in ensembles.items():
    print(f"\n   >>> Training Ensemble Engine: {name} ...")
    
    # Train & Predict
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    # Evaluate
    g_mcc = matthews_corrcoef(y_test, preds)
    per_class_mcc = calculate_per_class_mcc(y_test, preds, CLASS_NAMES)
    
    # Calculate Custom MOSS Score
    moss_score = (g_mcc + per_class_mcc.get('TRI', 0.0) + per_class_mcc.get('TET', 0.0)) / 3
    
    print(f"       Global MCC: {g_mcc:.4f} | MOSS Score: {moss_score:.4f}")
    
    results_log.append({
        "Ensemble_Type": name,
        "MOSS_Score": moss_score,
        "Global_MCC": g_mcc,
        "PD_MCC": per_class_mcc.get('PD', 0.0),
        "APD_MCC": per_class_mcc.get('APD', 0.0),
        "TRI_MCC": per_class_mcc.get('TRI', 0.0),
        "TET_MCC": per_class_mcc.get('TET', 0.0)
    })

# ==============================================================================
# 7. FINAL COMPARISON
# ==============================================================================
print("\n" + "="*100)
print("🏆 MULTICLASS ENSEMBLE RESULTS COMPARISON (Full RBF Dataset) 🏆")
print("="*100)

# Create DataFrame & Sort by Custom MOSS Score
df_res = pd.DataFrame(results_log).sort_values(by="MOSS_Score", ascending=False)

print(df_res.to_string(index=False, float_format="%.4f"))

# Save Outputs
out_file = os.path.join(BASE_DIR, "07_Multiclass_Ensemble_Results.csv")
df_res.to_csv(out_file, index=False)
print(f"\n✅ Results successfully generated and exported to:\n{out_file}")