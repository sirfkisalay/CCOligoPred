"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 16:  Mutagenesis Generator
==============================================================================
Description:
Generates in-silico point mutations based on heptad register positions 
(a, b, c, d, e, f, g) for downstream structural validation using AlphaFold 3.

Mutation Strategies:
- MutA: Destabilize hydrophobic 'a' core (Val/Ile -> Asn).
- MutB: Destabilize 'g-a' inter-chain electrostatic lock (Arg-Val -> Ala-Val).
- MutC: Enforce C-terminal steric clash at 'e' position (Native -> Trp).
- MutD: Overload the middle 'a' position (Native -> Phe).

Inputs: 'Register_anotation_CCdb_MARCOIL.xlsx'
Outputs: 'AlphaFold3_Validation_Mutants.fasta'
==============================================================================
"""

import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & DYNAMIC PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS", "validation")

# Ensure the validation directory exists
os.makedirs(DATA_DIR, exist_ok=True)

ANNOTATION_FILE = os.path.join(DATA_DIR, "Register_anotation_CCdb_MARCOIL.xlsx")
OUTPUT_FASTA = os.path.join(DATA_DIR, "AlphaFold3_Validation_Mutants.fasta")

# ==============================================================================
# 2. MUTATION LOGIC
# ==============================================================================
def generate_mutants(sequence, register):
    """
    Generates targeted mutants based on sequence and register alignment.
    """
    seq = str(sequence).strip()
    reg = str(register).replace(" ", "").strip()
    
    min_len = min(len(seq), len(reg))
    seq = seq[:min_len]
    reg = reg[:min_len]
    
    length = len(seq)
    if length < 9: # Skip if sequence is too short to have proper segments
        return {}

    mid_start = length // 3
    c_start = 2 * (length // 3)
    
    mutants = {}
    
    # MutA: Break the hydrophobic 'a' core (Val/Ile -> Asn)
    seq_a = list(seq)
    mutated_a = False
    for i in range(length):
        if reg[i] == 'a' and seq_a[i] in ['V', 'I']:
            seq_a[i] = 'N' 
            mutated_a = True
    if mutated_a:
        mutants['MutA_Destabilize_a_Core'] = "".join(seq_a)

    # MutB: Break the 'g-a' inter-chain lock (Arg-Val -> Ala-Val)
    seq_b = list(seq)
    mutated_b = False
    for i in range(length - 1):
        if reg[i] == 'g' and reg[i+1] == 'a':
            if seq_b[i] == 'R' and seq_b[i+1] == 'V':
                seq_b[i] = 'A'
                mutated_b = True
    if mutated_b:
        mutants['MutB_Destabilize_ga_Lock'] = "".join(seq_b)

    # MutC: Force C-terminal Steric Clash at 'e' (Native -> Trp)
    seq_c = list(seq)
    mutated_c = False
    for i in range(c_start, length):
        if reg[i] == 'e' and seq_c[i] not in ['W', 'Y', 'F']:
            seq_c[i] = 'W' 
            mutated_c = True
    if mutated_c:
        mutants['MutC_Enforce_Cterm_e_Clash'] = "".join(seq_c)

    # MutD: Overload the Middle 'a' position (Native -> Phe)
    seq_d = list(seq)
    mutated_d = False
    for i in range(mid_start, c_start):
        if reg[i] == 'a' and seq_d[i] not in ['F', 'W', 'Y']:
            seq_d[i] = 'F' 
            mutated_d = True
    if mutated_d:
        mutants['MutD_Enforce_Mid_a_Overload'] = "".join(seq_d)

    return mutants

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
print(f"[*] Loading sequence and register data from {DATA_DIR}...")

try:
    if ANNOTATION_FILE.endswith('.csv'):
        df = pd.read_csv(ANNOTATION_FILE)
    else:
        df = pd.read_excel(ANNOTATION_FILE)
except FileNotFoundError:
    print(f"[!] ERROR: Could not find the file at:\n{ANNOTATION_FILE}")
    print("Please ensure 'Register_anotation_CCdb_MARCOIL.xlsx' is placed in the DATASETS/validation folder.")
    exit()

print(f" -> ✅ File loaded successfully! Total rows found: {len(df)}")

# Fuzzy Column Matching (To catch typos like 'anotation' vs 'annotation')
pdb_col = next((col for col in df.columns if 'PDB' in str(col).upper()), None)
seq_col = next((col for col in df.columns if 'SEQ' in str(col).upper()), None)
reg_col = next((col for col in df.columns if 'HEPTAD' in str(col).upper() or 'REG' in str(col).upper()), None)

print(f"\n[*] Mapping columns:")
print(f" - PDB Name mapped to : '{pdb_col}'")
print(f" - Sequence mapped to : '{seq_col}'")
print(f" - Register mapped to : '{reg_col}'")

if not all([pdb_col, seq_col, reg_col]):
    print("\n[!] ERROR: Could not map all necessary columns. Please check your Excel headers.")
    exit()

# ==============================================================================
# 4. MUTANT GENERATION & EXPORT
# ==============================================================================
total_wt = 0
total_mutants = 0

with open(OUTPUT_FASTA, 'w') as f:
    for index, row in df.iterrows():
        pdb_name = str(row[pdb_col]).strip()
        wt_seq = str(row[seq_col]).strip()
        reg_str = str(row[reg_col]).strip()
        
        # Skip garbage rows/empty data
        if pd.isna(row[seq_col]) or wt_seq == 'nan' or len(wt_seq) < 5:
            continue
            
        # Write Wild-Type sequence
        f.write(f">{pdb_name}_WildType\n{wt_seq}\n")
        total_wt += 1
        
        # Write Mutants
        mutants = generate_mutants(wt_seq, reg_str)
        for mut_name, mut_seq in mutants.items():
            f.write(f">{pdb_name}_{mut_name}\n{mut_seq}\n")
            total_mutants += 1

print("\n" + "="*80)
print("✅ MUTAGENESIS FASTA GENERATION COMPLETE")
print("="*80)
print(f"Processed Wild-Type Sequences : {total_wt}")
print(f"Generated Mutant Sequences    : {total_mutants}")
print(f"\nFile successfully saved to:\n{OUTPUT_FASTA}")