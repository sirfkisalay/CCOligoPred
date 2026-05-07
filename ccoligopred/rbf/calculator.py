import pandas as pd
import numpy as np
from collections import Counter
import itertools

# --- GLOBAL CONSTANTS ---
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

PROPERTY_SCALES = {
    'hydrophobicity_kd': {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2},
    'hydrophobicity_eisenberg': {'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29, 'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38, 'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12, 'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08},
    'polarity_grantham': {'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5, 'Q': 10.5, 'E': 12.3, 'G': 9.0, 'H': 10.4, 'I': 5.2, 'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0, 'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2, 'V': 5.9},
    'molecular_weight': {'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16, 'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.17, 'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13, 'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15},
    'bulkiness_zimmerman': {'A': 11.5, 'R': 14.28, 'N': 12.82, 'D': 11.68, 'C': 13.46, 'Q': 14.45, 'E': 13.57, 'G': 3.4, 'H': 13.69, 'I': 21.4, 'L': 21.4, 'K': 15.71, 'M': 16.25, 'F': 19.8, 'P': 17.43, 'S': 9.47, 'T': 15.77, 'W': 21.67, 'Y': 18.03, 'V': 21.57},
    'charge': {'D':-1, 'E':-1, 'K':1, 'R':1, 'H':0.1, 'A':0, 'N':0, 'C':0, 'Q':0, 'G':0, 'I':0, 'L':0, 'M':0, 'F':0, 'P':0, 'S':0, 'T':0, 'W':0, 'Y':0, 'V':0},
    'alpha_helix_propensity': {'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70, 'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08, 'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57, 'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06},
    'sheet_propensity_chou': {'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19, 'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60, 'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55, 'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70},
    'turn_propensity_chou': {'A': 0.66, 'R': 0.95, 'N': 1.56, 'D': 1.46, 'C': 1.19, 'Q': 0.98, 'E': 0.74, 'G': 1.56, 'H': 0.95, 'I': 0.47, 'L': 0.59, 'K': 1.01, 'M': 0.60, 'F': 0.60, 'P': 1.52, 'S': 1.43, 'T': 0.96, 'W': 0.96, 'Y': 1.14, 'V': 0.50}
}

GROUP_DEFINITIONS = {
    'hydrophobic': {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'G', 'P', 'C', 'Y'},
    'aliphatic': {'A', 'V', 'I', 'L'},
    'aromatic': {'F', 'W', 'Y', 'H'},
    'polar_uncharged': {'S', 'T', 'N', 'Q', 'Y'},
    'charged': {'D', 'E', 'K', 'R', 'H'},
    'positive': {'K', 'R', 'H'},
    'negative': {'D', 'E'},
    'tiny': {'A', 'G', 'C', 'S'},
    'large': {'L', 'I', 'F', 'W', 'Y', 'R', 'K', 'M', 'E', 'Q'}
}
GROUP_ORDER = list(GROUP_DEFINITIONS.keys())

# --- HELPER FUNCTIONS ---
def get_positional_residues(sequence, register):
    sequence, register = str(sequence), str(register)
    min_len = min(len(sequence), len(register))
    positional_residues = {pos: [] for pos in 'abcdefg'}
    for aa, pos in zip(sequence[:min_len], register[:min_len]):
        if aa in AMINO_ACIDS:
            positional_residues[pos].append(aa)
    return positional_residues

def calculate_enhanced_physicochem(sequence, register):
    positional_residues = get_positional_residues(sequence, register)
    feature_vector = []
    for pos in 'abcdefg':
        residues = positional_residues[pos]
        for scale in PROPERTY_SCALES.values():
            if not residues:
                feature_vector.extend([0, 0])
                continue
            values = [scale.get(aa, 0) for aa in residues]
            feature_vector.extend([np.mean(values), np.std(values)])
    return feature_vector

def calculate_register_pair_composition(sequence, register):
    sequence, register = str(sequence), str(register)
    features = {}
    pair_types = {'ad': ('a', 'd', 3), 'eg': ('e', 'g', 2), 'ga_prime': ('g', 'a', 1)}
    possible_pairs = [''.join(p) for p in itertools.product(AMINO_ACIDS, repeat=2)]

    for pair_name, (pos1, pos2, offset) in pair_types.items():
        pair_counts = Counter()
        for i in range(len(sequence) - offset):
            if i + offset < len(register) and register[i] == pos1 and register[i+offset] == pos2:
                aa1, aa2 = sequence[i], sequence[i+offset]
                if aa1 in AMINO_ACIDS and aa2 in AMINO_ACIDS:
                    pair_counts[aa1 + aa2] += 1
        
        total_pairs = sum(pair_counts.values())
        if total_pairs > 0:
            for pair in possible_pairs:
                features[f'Pair_{pair}_{pair_name}'] = pair_counts[pair] / total_pairs
        else:
            for pair in possible_pairs:
                features[f'Pair_{pair}_{pair_name}'] = 0
    return features

def calculate_positional_aac(sequence, register):
    positional_residues = get_positional_residues(sequence, register)
    feature_vector = []
    for pos in 'abcdefg':
        counts = Counter(positional_residues[pos])
        total = len(positional_residues[pos])
        for aa in AMINO_ACIDS:
            feature_vector.append(counts[aa] / total if total > 0 else 0)
    return feature_vector

def calculate_base_physicochemical_profile(sequence, register):
    positional_residues = get_positional_residues(sequence, register)
    feature_vector = []
    base_props = ['hydrophobicity_kd', 'charge', 'molecular_weight']
    for pos in 'abcdefg':
        residues = positional_residues[pos]
        if not residues:
            feature_vector.extend([0, 0, 0])
            continue
        feature_vector.extend([np.mean([PROPERTY_SCALES[prop].get(aa, 0) for aa in residues]) for prop in base_props])
    return feature_vector

def calculate_grouped_features(sequence, register):
    positional_residues = get_positional_residues(sequence, register)
    feature_vector = []
    for pos in 'abcdefg':
        residues = positional_residues.get(pos, [])
        total = len(residues)
        if total == 0:
            feature_vector.extend([0] * len(GROUP_ORDER))
            continue
        for group_name in GROUP_ORDER:
            count = sum(1 for aa in residues if aa in GROUP_DEFINITIONS[group_name])
            feature_vector.append(count / total)
    return feature_vector

def generate_segmented_features(sequence, register):
    sequence, register = str(sequence), str(register)
    full_feature_vector = []
    seq_len = min(len(sequence), len(register))
    
    seg1_end = seq_len // 3
    seg2_end = 2 * (seq_len // 3)
    
    segments = [
        (sequence[:seg1_end], register[:seg1_end]),
        (sequence[seg1_end:seg2_end], register[seg1_end:seg2_end]),
        (sequence[seg2_end:], register[seg2_end:])
    ]
    
    num_feats_per_segment = 140 + 21 + (len(GROUP_ORDER) * 7)

    for (seq_seg, reg_seg) in segments:
        if not seq_seg:
            full_feature_vector.extend([0] * num_feats_per_segment)
            continue
        full_feature_vector.extend(calculate_positional_aac(seq_seg, reg_seg))
        full_feature_vector.extend(calculate_base_physicochemical_profile(seq_seg, reg_seg))
        full_feature_vector.extend(calculate_grouped_features(seq_seg, reg_seg))
    return full_feature_vector

def calculate_register_autocorrelation(sequence, register):
    positional_residues = get_positional_residues(sequence, register)
    feature_vector = []
    props_to_use = {'hydrophobicity_kd': PROPERTY_SCALES['hydrophobicity_kd'], 'alpha_helix_propensity': PROPERTY_SCALES['alpha_helix_propensity']}
    max_lag = 3
    
    for pos in 'abcdefg':
        residues = positional_residues[pos]
        for prop_name, scale in props_to_use.items():
            prop_seq = [scale.get(aa, 0) for aa in residues]
            n = len(prop_seq)
            prop_mean = np.mean(prop_seq) if n > 0 else 0
            prop_std = np.std(prop_seq) if n > 1 else 1.0
            if prop_std == 0: prop_std = 1.0
            
            norm_prop_seq = [(p - prop_mean) / prop_std for p in prop_seq]

            for lag in range(1, max_lag + 1):
                if n > lag:
                    ac_sum = sum(norm_prop_seq[i] * norm_prop_seq[i+lag] for i in range(n - lag))
                    feature_vector.append(ac_sum / (n - lag))
                else:
                    feature_vector.append(0)
    return feature_vector

# --- MAIN RBF GENERATOR ---
def generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with 'Sequence' and 'Register' columns and returns 
    a DataFrame containing the complete calculated RBF feature matrix.
    """
    # Create internal copy to ensure lowercase matching
    data_df = df.copy()
    data_df.columns = [col.lower() for col in data_df.columns]
    
    all_feature_dfs = []
    
    # 1. Physicochemical
    phys_cols = [f"{stat}_{prop}_{pos}" for pos in 'abcdefg' for prop in PROPERTY_SCALES.keys() for stat in ['Mean', 'Std']]
    phys_features = data_df.apply(lambda row: calculate_enhanced_physicochem(row['sequence'], row['register']), axis=1)
    phys_df = pd.DataFrame(phys_features.tolist(), columns=phys_cols, index=data_df.index)
    all_feature_dfs.append(phys_df)
    
    # 2. Pair Composition
    pair_features_list = data_df.apply(lambda row: calculate_register_pair_composition(row['sequence'], row['register']), axis=1)
    pair_df = pd.DataFrame(pair_features_list.tolist(), index=data_df.index)
    all_feature_dfs.append(pair_df)
    
    # 3. Segments
    base_aac_cols = [f"AAC_{aa}_pos_{pos}" for pos in 'abcdefg' for aa in AMINO_ACIDS]
    base_phys_cols = [f"Mean{prop}_{pos}" for pos in 'abcdefg' for prop in ['Hydro', 'Charge', 'MW']]
    base_group_cols = [f"Group_{group}_{pos}" for pos in 'abcdefg' for group in GROUP_ORDER]
    seg_cols = [f"{col}_{seg}" for seg in ['N_term', 'Middle', 'C_term'] for col in base_aac_cols + base_phys_cols + base_group_cols]
    seg_features = data_df.apply(lambda row: generate_segmented_features(row['sequence'], row['register']), axis=1)
    seg_df = pd.DataFrame(seg_features.tolist(), columns=seg_cols, index=data_df.index)
    all_feature_dfs.append(seg_df)
    
    # 4. Autocorrelation
    ac_cols = [f"AC_{prop}_lag{lag}_{pos}" for pos in 'abcdefg' for prop in ['hydrophobicity_kd', 'alpha_helix_propensity'] for lag in range(1, 4)]
    ac_features = data_df.apply(lambda row: calculate_register_autocorrelation(row['sequence'], row['register']), axis=1)
    ac_df = pd.DataFrame(ac_features.tolist(), columns=ac_cols, index=data_df.index)
    all_feature_dfs.append(ac_df)
    
    # Combine everything
    final_features_df = pd.concat(all_feature_dfs, axis=1)
    return final_features_df
