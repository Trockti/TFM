"""
Simple script to evaluate models on the MUCH benchmark in Spanish.
"""

import os
from collections import defaultdict
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Change to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from much.src.annotation import get_all_annotations
from much.src.utils.constants import LLMS_INFOS

# Configuration
LANG = "es"  # Spanish only
# Use open-source models as ground truth annotators
ANNOTATORS = ["gemma3_12b", "gemma3_27b"]  # Must agree to be used as ground truth

print("="*80)
print(f"Evaluating models on MUCH benchmark - Spanish ({LANG})")
print(f"Using {ANNOTATORS[0]} and {ANNOTATORS[1]} as ground truth annotators")
print("="*80)

# Load annotations
print("\nLoading annotations...")
all_annotations = get_all_annotations(reload=True)

# Filter for Spanish and samples where annotators agree
annotations_filtered = {
    key: value
    for key, value in all_annotations.items()
    if value.generation.lang == LANG 
    and all(annotator in value.labels for annotator in ANNOTATORS)
    and value.labels[ANNOTATORS[0]] == value.labels[ANNOTATORS[1]]
}

print(f"Found {len(annotations_filtered)} Spanish samples with agreed annotations")

# Count statistics per model
model_stats = defaultdict(lambda: {
    "samples": 0,
    "total_claims": 0,
    "hallucinated_claims": 0,
    "samples_with_hallucinations": 0,
})

for annotation in annotations_filtered.values():
    model_name = annotation.generation.generation_cfg.model_name
    labels = annotation.labels[ANNOTATORS[0]]  # Ground truth
    
    model_stats[model_name]["samples"] += 1
    model_stats[model_name]["total_claims"] += len(labels) - 1  # Exclude EOS token
    
    if -1 in labels:  # Has hallucinations
        model_stats[model_name]["hallucinated_claims"] += np.sum(np.array(labels[:-1]) == -1)
        model_stats[model_name]["samples_with_hallucinations"] += 1

# Print results
print("\n" + "="*80)
print("Model Performance Statistics (Spanish only, no trash split)")
print("="*80)

for model_name in sorted(model_stats.keys()):
    stats = model_stats[model_name]
    if stats["samples"] == 0:
        continue
    
    hallucination_rate = stats["hallucinated_claims"] / stats["total_claims"]
    sample_halluc_rate = stats["samples_with_hallucinations"] / stats["samples"]
    
    print(f"\n{model_name}:")
    print(f"  Samples: {stats['samples']}")
    print(f"  Total claims: {stats['total_claims']}")
    print(f"  Hallucinated claims: {stats['hallucinated_claims']} ({hallucination_rate:.2%})")
    print(f"  Samples with hallucinations: {stats['samples_with_hallucinations']} ({sample_halluc_rate:.2%})")

print("\n" + "="*80)
