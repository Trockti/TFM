import os
import json
import glob
from collections import defaultdict

def compute_readability_differences():
    # Nested dictionary to store score differences
    # Structure: results[folder_name][model_name] = {'fh_diffs': [], 'inflesz_diffs': []}
    results = defaultdict(lambda: defaultdict(lambda: {'fh_diffs': [], 'inflesz_diffs': []}))

    # Look for all metrics_*.json files inside any folder starting with 'results_'
    file_paths = glob.glob("results_*/metrics_*.json")

    if not file_paths:
        print("No files matching 'results_*/metrics_*.json' were found.")
        return

    for file_path in file_paths:
        # Extract the folder name (e.g., 'results_v1')
        folder_name = os.path.dirname(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading JSON from {file_path}. Skipping.")
                continue
            
        for item in data:
            # Safely get the model name, default to unknown if missing
            model_name = item.get("model_name", "unknown")
            
            try:
                # Extract Fernandez-Huerta scores
                fh_orig = item["scores"]["readability"]["Fernandez-Huerta"]["original"]["fernandez_huerta"]
                fh_simp = item["scores"]["readability"]["Fernandez-Huerta"]["simplified"]["fernandez_huerta"]
                
                # Extract Inflesz scores
                inflesz_orig = item["scores"]["readability"]["Inflesz"]["original"]["inflesz"]
                inflesz_simp = item["scores"]["readability"]["Inflesz"]["simplified"]["inflesz"]
                
                # Append differences (Simplified - Original)
                results[folder_name][model_name]['fh_diffs'].append(fh_simp - fh_orig)
                results[folder_name][model_name]['inflesz_diffs'].append(inflesz_simp - inflesz_orig)
                
            except KeyError as e:
                # Skip items that don't match the expected schema
                # print(f"Missing key {e} in {file_path}")
                continue

    # Calculate and display the results
    print("=== Readability Mean Differences (Simplified - Original) ===\n")
    
    # Sort folders to output in logical order (results_v1, results_v2, etc.)
    for folder in sorted(results.keys()):
        print(f"📁 {folder}")
        
        for model, diffs in results[folder].items():
            fh_list = diffs['fh_diffs']
            inflesz_list = diffs['inflesz_diffs']
            
            # Calculate means
            mean_fh = sum(fh_list) / len(fh_list) if fh_list else 0
            mean_inflesz = sum(inflesz_list) / len(inflesz_list) if inflesz_list else 0
            
            # Formatting with a '+' sign for positive numbers
            print(f"  └─ Model: {model}")
            print(f"       Mean Fernandez-Huerta Diff: {mean_fh:+.4f}")
            print(f"       Mean Inflesz Diff:          {mean_inflesz:+.4f}")
        print() # blank line for readability

if __name__ == "__main__":
    compute_readability_differences()