import argparse
import sys
import warnings

# Suppress standard warnings to keep the terminal output clean for the end-user
warnings.filterwarnings("ignore")

# Import the modules we built
from oligocoil.io.excel_handler import load_sequence_data, save_predictions
from oligocoil.rbf import generate_all_features
from oligocoil.models import predict_multiclass, predict_binary

def main():
    # Set up the command-line arguments
    parser = argparse.ArgumentParser(
        description="OligoCoil Predictor: A dual-framework machine learning tool for predicting coiled-coil oligomeric states."
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input Excel file containing 'Sequence' and 'Register' columns."
    )
    
    parser.add_argument(
        "-o", "--output", 
        default="oligocoil_predictions.xlsx", 
        help="Path to save the output Excel file (default: oligocoil_predictions.xlsx)."
    )
    
    args = parser.parse_args()
    
    print("="*65)
    print(" O L I G O C O I L   P R E D I C T O R   (v1.0.0)")
    print("="*65)
    
    try:
        # 1. Load Data
        print(f"[*] Loading input data from: {args.input}")
        df = load_sequence_data(args.input)
        print(f"    -> Successfully loaded {len(df)} sequences.")
        
        # 2. Calculate Features
        print("[*] Generating Register-Based Features (RBF)...")
        feature_matrix = generate_all_features(df)
        print(f"    -> Feature extraction complete. Matrix shape: {feature_matrix.shape}")
        
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
