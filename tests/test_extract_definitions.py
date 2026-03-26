import pytest
import json
from unittest.mock import MagicMock, patch
import sys
import os


# Import the module to be tested
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auxiliar_programs import extract_definitions


def test_definitions_cervantes_analysis():
    """
    Checks if definitions in test_definitions_cervantes.json are included in extracted_definitions_cervantes_inc.json
    and if they match. Returns/Prints missing words and failure rates.
    """
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ground_truth_file = os.path.join(current_dir, 'test_definitions_cervantes.json')
    extracted_file = os.path.join(current_dir, '../data/extracted_definitions_cervantes_inc.json')
    
    # Load Expected Definitions
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
        
    # Load Extracted Data
    if not os.path.exists(extracted_file):
        pytest.fail(f"Extracted file not found: {extracted_file}")
        
    with open(extracted_file, 'r', encoding='utf-8') as f:
        extracted = json.load(f)
        
    # Prepare comparison dictionary
    extracted_lower = {k.lower().strip(): v for k, v in extracted.items()}
    
    missing_words = []
    wrong_definitions = []
    total_words = len(ground_truth)
    
    for word, expected_def in ground_truth.items():
        # Clean up the key for lookup
        word_key = word.lower().strip()
        
        if word_key not in extracted_lower:
            missing_words.append(word)
        else:
            actual_def = extracted_lower[word_key]
            # Compare definitions (normalizing stripping whitespace)
            if actual_def.strip() != expected_def.strip():
                wrong_definitions.append({
                    'word': word,
                    'expected': expected_def,
                    'actual': actual_def
                })
                
    # Calculate requested metrics
    missing_count = len(missing_words)
    wrong_def_count = len(wrong_definitions)
    total_failures = missing_count + wrong_def_count
    
    # Percentage of failures (words missing + definitions wrong)
    failure_rate = (total_failures / total_words * 100) if total_words > 0 else 0
    
    # Percentage of wrong definitions (relative to total words checked)
    wrong_def_rate = (wrong_def_count / total_words * 100) if total_words > 0 else 0
    
    # Print results (Use 'pytest -s' to see this output)
    print(f"\n--- Cervantes Definitions Analysis ---")
    print(f"Total Words Checked: {total_words}")
    print(f"Words Missing: {missing_words}")
    print(f"Failure Rate (Missing + Wrong Defs): {failure_rate:.2f}%")
    print(f"Wrong Definition Rate: {wrong_def_rate:.2f}%")
    print(f"--------------------------------------")

def test_definitions_damaso_analysis():
    """
    Checks if definitions in test_definitions_damaso.json are included in extracted_definitions_damaso.json
    and if they match. Returns/Prints missing words and failure rates.
    """
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ground_truth_file = os.path.join(current_dir, 'test_definitions_damaso.json')
    extracted_file = os.path.join(current_dir, '../data/extracted_definitions_damaso.json')
    
    # Load Expected Definitions
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
        
    # Load Extracted Data
    if not os.path.exists(extracted_file):
        pytest.fail(f"Extracted file not found: {extracted_file}")
        
    with open(extracted_file, 'r', encoding='utf-8') as f:
        extracted = json.load(f)
        
    # Prepare comparison dictionary
    extracted_lower = {k.lower().strip(): v for k, v in extracted.items()}
    
    missing_words = []
    wrong_definitions = []
    total_words = len(ground_truth)
    
    for word, expected_def in ground_truth.items():
        # Clean up the key for lookup
        word_key = word.lower().strip()
        
        if word_key not in extracted_lower:
            missing_words.append(word)
        else:
            actual_def = extracted_lower[word_key]
            # Compare definitions (normalizing stripping whitespace)
            if actual_def.strip() != expected_def.strip():
                wrong_definitions.append({
                    'word': word,
                    'expected': expected_def,
                    'actual': actual_def
                })
                
    # Calculate requested metrics
    missing_count = len(missing_words)
    wrong_def_count = len(wrong_definitions)
    total_failures = missing_count + wrong_def_count
    
    # Percentage of failures (words missing + definitions wrong)
    failure_rate = (total_failures / total_words * 100) if total_words > 0 else 0
    
    # Percentage of wrong definitions (relative to total words checked)
    wrong_def_rate = (wrong_def_count / total_words * 100) if total_words > 0 else 0
    
    # Print results (Use 'pytest -s' to see this output)
    print(f"\n--- Damaso Definitions Analysis ---")
    print(f"Total Words Checked: {total_words}")
    print(f"Words Missing: {missing_words}")
    print(f"Failure Rate (Missing + Wrong Defs): {failure_rate:.2f}%")
    print(f"Wrong Definition Rate: {wrong_def_rate:.2f}%")
    print(f"--------------------------------------")