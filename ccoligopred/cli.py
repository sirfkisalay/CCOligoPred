import argparse
import sys
import warnings
import pandas as pd  # Added to handle saving the standalone feature matrix

# Suppress standard warnings to keep the terminal output clean for the end-user
warnings.filterwarnings("ignore")

# Import the modules 
from ccoligopred.io.excel_handler import load_sequence_data, save_predictions
from ccoligopred.rbf import generate_all_features
from ccoligopred.models import predict_multiclass, predict_binary

def main():
    # Set up the main command-line parser
    parser = argparse.ArgumentParser(
        description="CCOligoPred: A dual-framework machine learning tool for coiled-coil prediction and RBF extraction."
    )
    
    # Create subparsers for the distinct commands
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # =========================================================================
    # COMMAND 1: FULL PREDICTION (ccoligopred predict)
    # =========================================================================
    parser_predict = subparsers.add_parser("predict", help="Run full multiclass and binary CCD state predictions.")
    parser_predict.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input Excel file containing 'Sequence' and 'Register' columns."
    )
    parser_predict.add_argument(
        "-o", "--output", 
        default="ccoligopred_predictions.xlsx", 
        help="Path to save the output Excel file (default: ccoligopred_predictions.xlsx)."
    )

    # =========================================================================
    # COMMAND 2: RBF EXTRACTION ONLY (ccoligopred rbf)
    # =========================================================================
    parser_rbf = subparsers.add_parser("rbf", help="Calculate and extract the 2041-dimensional Register-Based Features (RBF) only.")
    parser_rbf.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input Excel file containing 'Sequence' and 'Register' columns."
    )
    parser_rbf.add_argument(
        "-o", "--output", 
        default="ccoligopred_rbf_features.xlsx", 
        help="Path to save the extracted RBF feature matrix (default: ccoligopred_rbf_features.xlsx)."
    )

    args = parser.parse_args()
    
    # =========================================================================
    # EXECUTION LOGIC
    # =========================================================================
    print("="*65)
    if args.command == "predict":
        print(" CCOligoPred   P R E D I C T O R   (v1.0.0)")
    elif args.command == "rbf":
        print(" CCOligoPred   R B F   E X T R A C T O R   (v1.0.0)")
    print("="*65)
    
    try:
        # 1. Load Data (Common step for both commands)
        print(f"[*] Loading input data from: {args.input}")
        df = load_sequence_data(args.input)
        print(f"    -> Successfully loaded {len(df)} sequences.")
        
        # 2. Calculate Features (Common step for both commands)
        print("[*] Generating Register-Based Features (RBF)...")
        feature_matrix = generate_all_features(df)
        print(f"    -> Feature extraction complete. Matrix shape: {feature_matrix.shape}")

        # ---------------------------------------------------------
        # ROUTE A: RBF ONLY MODE
        # ---------------------------------------------------------
        if args.command == "rbf":
            print("[*] Bypassing predictive models.")
            print(f"[*] Saving feature matrix to: {args.output}")
            
            # Assuming feature_matrix is a DataFrame. We combine it with the original sequence data
            if isinstance(feature_matrix, pd.DataFrame):
                final_rbf_df = pd.concat([df, feature_matrix], axis=1)
            else:
                # Fallback just in case your generate_all_features returns a numpy array
                feature_df = pd.DataFrame(feature_matrix)
                final_rbf_df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
                
            final_rbf_df.to_excel(args.output, index=False)
            
            print("="*65)
            print("*** RBF EXTRACTION COMPLETE! ***")
            print(f"Your feature matrix is saved at: {args.output}")
            print("="*65)
            return

        # ---------------------------------------------------------
        # ROUTE B: FULL PREDICTION MODE
        # ---------------------------------------------------------
        elif args.command == "predict":
            # 3. Binary Prediction
            print("[*] Running Binary Classifier (Trimer vs Non-Trimer)...")
            binary_preds, binary_probs = predict_binary(feature_matrix)
            
            # 4. Multiclass Prediction
            print("[*] Running Multiclass Predictor (PD, APD, TRI, TETRA)...")
            multiclass_preds = predict_multiclass(feature_matrix)
            
            # 5. Save Output
            print(f"[*] Saving predictions to: {args.output}")
            save_predictions(df, multiclass_preds, binary_preds, binary_probs, args.output)
            
            print("="*65)
            print("*** PREDICTION COMPLETE! ***")
            print(f"Your results are saved at: {args.output}")
            print("="*65)
        
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()