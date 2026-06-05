"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 15: Final Binary Model Extraction & Explainable AI (SHAP)
==============================================================================
Description:
1. Trains the finalized Champion Stacking Ensemble (LGBM, XGB, RF -> LogReg) 
   using the Full RBF dataset for binary classification (Target: State_tri).
2. Evaluates performance, plots Confusion Matrix and ROC curve, and saves the 
   model artifact (.pkl).
3. Executes Explainable AI (XAI) using SHAP KernelExplainer to extract the top 
   biological feature drivers promoting True Positives (Trimers) and True 
   Negatives (Non-Trimers/Rest).

Outputs:
- 15_Final_Binary_Champion_Stacking.pkl
- 15_Binary_Champion_Confusion_Matrix.png & 15_Binary_Champion_ROC_Curve.png
- 15_Drivers_of_Trimers_TP.csv & 15_Drivers_of_NonTrimers_TN.csv
- 15_SHAP_Plot_TP_Trimer.png & 15_SHAP_Plot_TN_NonTrimer.png
==============================================================================
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Metrics
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Models
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb

# Explainable AI
import shap

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS", "binary")

TRAIN_FILE = "BINARY_RBF_TRAIN.xlsx"
TEST_FILE  = "BINARY_RBF_TEST.xlsx"
TARGET_COLUMN = "State_tri"
CLASS_NAMES = ['Rest', 'Tri']
RANDOM_STATE = 42

# --- OPTIMIZED HYPERPARAMETERS ---
rf_p = {'n_estimators': 1300, 'max_depth': 23, 'min_samples_leaf': 9, 'max_features': 0.5, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': 0}
xgb_p = {'n_estimators': 1150, 'max_depth': 12, 'learning_rate': 0.0405246, 'subsample': 0.75433, 'colsample_bytree': 0.60561, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}
lgb_p = {'n_estimators': 860, 'learning_rate': 0.046423, 'num_leaves': 30, 'max_depth': 13, 'subsample': 0.80635, 'colsample_bytree': 0.67299, 'reg_alpha': 1.1078e-08, 'reg_lambda': 0.17206, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1, 'objective': 'binary'}

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Binary)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 3. DATA LOADING
# ==============================================================================
print(f"[*] Loading Datasets from {DATA_DIR}...")
train_df = pd.read_excel(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_excel(os.path.join(DATA_DIR, TEST_FILE))

y_train = train_df[TARGET_COLUMN].astype(int)
y_test  = test_df[TARGET_COLUMN].astype(int)
X_train = train_df.drop(columns=[TARGET_COLUMN])
X_test  = test_df.drop(columns=[TARGET_COLUMN])

X_test = X_test[X_train.columns] # Ensure alignment
feature_names = X_train.columns.tolist()

print(f" -> Data Loaded. Training on {X_train.shape[0]} samples, {X_train.shape[1]} features.\n")

# ==============================================================================
# 4. ENSEMBLE CONSTRUCTION & TRAINING
# ==============================================================================
print("[*] Building & Training Stacking Architecture (LGBM+XGB+RF -> LogReg)...")

# Base Pipelines (Leakage Proof)
pipe_lgb = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('clf', lgb.LGBMClassifier(**lgb_p))])
pipe_xgb = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('clf', xgb.XGBClassifier(**xgb_p))])
pipe_rf  = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('clf', RandomForestClassifier(**rf_p))])

final_stack = StackingClassifier(
    estimators=[('lgb', pipe_lgb), ('xgb', pipe_xgb), ('rf', pipe_rf)],
    final_estimator=LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE),
    cv=5, n_jobs=1, passthrough=False 
)

final_stack.fit(X_train, y_train)

# ==============================================================================
# 5. VERIFICATION & PLOTTING
# ==============================================================================
print("[*] Verifying Performance on Test Set...")

preds_test = final_stack.predict(X_test)
probs_test = final_stack.predict_proba(X_test)[:, 1]

global_mcc = matthews_corrcoef(y_test, preds_test)
print(f"\n   >>> FINAL GLOBAL MCC: {global_mcc:.4f} <<<")
print(classification_report(y_test, preds_test, target_names=CLASS_NAMES, digits=4))

# Plots
print("[*] Generating Artifacts...")
plot_confusion_matrix(y_test, preds_test, CLASS_NAMES, f"Confusion Matrix (MCC={global_mcc:.3f})", os.path.join(BASE_DIR, "15_Binary_Champion_Confusion_Matrix.png"))
plot_roc_curve(y_test, probs_test, os.path.join(BASE_DIR, "15_Binary_Champion_ROC_Curve.png"))

# Save Model
artifact = {
    "model": final_stack,
    "feature_names": feature_names,
    "class_names": CLASS_NAMES,
    "model_score": global_mcc,
    "configuration": "Stacking (LGB+XGB+RF -> LogReg) | Full Features"
}
with open(os.path.join(BASE_DIR, "15_Final_Binary_Champion_Stacking.pkl"), "wb") as f:
    pickle.dump(artifact, f)

# ==============================================================================
# 6. EXPLAINABLE AI (XAI) - SHAP FEATURE DRIVERS
# ==============================================================================
print("\n" + "="*80)
print(" STARTING SHAP ANALYSIS FOR FEATURE DRIVERS ")
print("="*80)

# Filter TP and TN
analysis_df = X_test.copy()
analysis_df['Actual'] = y_test.values
analysis_df['Predicted'] = preds_test

tp_df = analysis_df[(analysis_df['Actual'] == 1) & (analysis_df['Predicted'] == 1)].drop(columns=['Actual', 'Predicted'])
tn_df = analysis_df[(analysis_df['Actual'] == 0) & (analysis_df['Predicted'] == 0)].drop(columns=['Actual', 'Predicted'])

print(f" -> Analyzing Drivers for {len(tp_df)} True Positives (Trimers)")
print(f" -> Analyzing Drivers for {len(tn_df)} True Negatives (Non-Trimers)")

# Initialize SHAP (K-Means background for speed)
background_data = shap.kmeans(X_train, 50)
predict_fn = lambda x: final_stack.predict_proba(x)[:, 1]
explainer = shap.KernelExplainer(predict_fn, background_data)

print("\n[*] Computing SHAP values (This may take a few minutes)...")
shap_values_tp = explainer.shap_values(tp_df)
shap_values_tn = explainer.shap_values(tn_df)

# Feature Extraction Function
def extract_feature_drivers(shap_vals, features, subset_name):
    mean_shap = np.mean(shap_vals, axis=0)
    driver_df = pd.DataFrame({'Feature': features, 'Mean_SHAP_Score': mean_shap, 'Abs_Score': np.abs(mean_shap)})
    driver_df = driver_df.sort_values(by='Abs_Score', ascending=False).reset_index(drop=True)
    
    if subset_name == "TP_Trimer":
        driver_df['Effect'] = driver_df['Mean_SHAP_Score'].apply(lambda x: "Promotes Trimerization" if x > 0 else "Hinders Trimerization")
    else:
        driver_df['Effect'] = driver_df['Mean_SHAP_Score'].apply(lambda x: "Promotes Non-Trimer (Stabilizes)" if x < 0 else "Promotes Trimer (Destabilizes)")
        
    return driver_df.drop(columns=['Abs_Score'])

# Extract and Save DataFrames
tp_drivers = extract_feature_drivers(shap_values_tp, feature_names, "TP_Trimer")
tn_drivers = extract_feature_drivers(shap_values_tn, feature_names, "TN_NonTrimer")

tp_drivers.to_csv(os.path.join(BASE_DIR, "15_Drivers_of_Trimers_TP.csv"), index=False)
tn_drivers.to_csv(os.path.join(BASE_DIR, "15_Drivers_of_NonTrimers_TN.csv"), index=False)

print("\nTOP 5 FEATURES DRIVING TRIMER PREDICTION (TP):")
print(tp_drivers.head(5))

# SHAP Visualizations
plt.figure()
plt.title("Key Features Driving Trimer Prediction (True Positives)")
shap.summary_plot(shap_values_tp, tp_df, show=False)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "15_SHAP_Plot_TP_Trimer.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
plt.title("Key Features Driving Non-Trimer Prediction (True Negatives)")
shap.summary_plot(shap_values_tn, tn_df, show=False)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "15_SHAP_Plot_TN_NonTrimer.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("✅ ALL TASKS COMPLETE! Artifacts, SHAP Plots, and CSVs saved successfully.")
print("="*80)