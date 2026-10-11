"""
biophysical_features.py
Computes 12 key physicochemical descriptors for antimicrobial peptide sequences.
"""

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Normalized Eisenberg consensus scale for hydrophobicity
EISENBERG_SCALE = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48,  'H': -0.40, 'I': 1.38,
    'L': 1.06,  'K': -1.50, 'M': 0.64,  'F': 1.19,  'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81,  'Y': 0.26,  'V': 1.08
}

def calculate_hydrophobic_moment(sequence, angle=100.0):
    """Calculates hydrophobic moment assuming an alpha-helical periodicity (100 deg)."""
    rad_angle = np.radians(angle)
    sin_sum = 0.0
    cos_sum = 0.0
    n = len(sequence)
    if n == 0:
        return 0.0
        
    for i, aa in enumerate(sequence):
        h = EISENBERG_SCALE.get(aa, 0.0)
        sin_sum += h * np.sin(i * rad_angle)
        cos_sum += h * np.cos(i * rad_angle)
        
    return np.sqrt(sin_sum**2 + cos_sum**2) / n

def calculate_boman_index(sequence):
    """Estimates protein-binding potential (Boman Index)."""
    boman_values = {
        'A': -0.44, 'R': -1.01, 'N': -0.22, 'D': -0.07, 'C': -0.13,
        'Q': -0.91, 'E': -0.67, 'G': -0.01, 'H': -0.17, 'I': -1.25,
        'L': -1.22, 'K': -0.67, 'M': -1.02, 'F': -1.52, 'P': -0.49,
        'S': -0.50, 'T': -0.77, 'W': -0.89, 'Y': -1.10, 'V': -1.15
    }
    if len(sequence) == 0:
        return 0.0
    return sum(boman_values.get(aa, 0.0) for aa in sequence) / len(sequence)

def extract_biophysical_features(sequence):
    """
    Extracts the 12 biophysical descriptors defined in thesis Section 4.5.3:
    1. Length
    2. Molecular Weight
    3. Net Charge (at pH 7.0)
    4. Charge Density
    5. Hydrophobicity (GRAVY)
    6. Hydrophobic Moment
    7. Aromatic Residue Fraction
    8. Aliphatic Residue Fraction
    9. Instability Index
    10. Isoelectric Point (pI)
    11. Boman Index
    12. Aliphatic Index / Composition metric
    """
    seq = sequence.strip().upper()
    analysis = ProteinAnalysis(seq)
    
    length = float(len(seq))
    mw = float(analysis.molecular_weight())
    net_charge = float(analysis.charge_at_pH(7.0))
    charge_density = net_charge / length if length > 0 else 0.0
    gravy = float(analysis.gravy())
    u_h = float(calculate_hydrophobic_moment(seq))
    aromaticity = float(analysis.aromaticity())
    
    # Aliphatic fraction: Alanine (A), Valine (V), Isoleucine (I), Leucine (L)
    aliphatic_count = sum(seq.count(aa) for aa in ['A', 'V', 'I', 'L'])
    aliphatic_fraction = float(aliphatic_count / length) if length > 0 else 0.0
    
    instability = float(analysis.instability_index())
    pi = float(analysis.isoelectric_point())
    boman = float(calculate_boman_index(seq))
    
    # Aliphatic index: X(Ala) + 2.9 * X(Val) + 3.9 * (X(Ile) + X(Leu))
    ala = seq.count('A') / length
    val = seq.count('V') / length
    ile = seq.count('I') / length
    leu = seq.count('L') / length
    aliphatic_index = float((ala + 2.9 * val + 3.9 * (ile + leu)) * 100.0)
    
    return np.array([
        length,
        mw,
        net_charge,
        charge_density,
        gravy,
        u_h,
        aromaticity,
        aliphatic_fraction,
        instability,
        pi,
        boman,
        aliphatic_index
    ], dtype=np.float32)
