import os
import glob
import json
import re
import nltk

# Ensure the sentence tokenizer model is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def analyze_context_metrics(base_path="."):
    """
    Scans through results_* folders, reads JSON files, and calculates 
    the mean characters, words, and sentences for the 'context' field.
    """
    # Dictionary to hold our final aggregated data
    # Format: stats[folder][model] = {means}
    stats = {}

    # Target all json files within folders starting with 'results_'
    search_pattern = os.path.join(base_path, "results_*", "*.json")
    json_files = glob.glob(search_pattern)

    if not json_files:
        print("No JSON files found in the specified path.")
        return

    for file_path in json_files:
        # Extract folder name (e.g., 'results_v2') and model name (e.g., 'latxa')
        folder_name = os.path.basename(os.path.dirname(file_path))
        model_name = os.path.basename(file_path).replace('.json', '')

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {file_path}. Skipping.")
                continue

        total_chars = 0
        total_words = 0
        total_sentences = 0
        valid_items_count = 0

        # Iterate through the JSON structure
        for item_id, item_data in data.items():
            context_text = item_data.get("context", "")
            
            # Skip empty or non-existent contexts
            if not context_text or not isinstance(context_text, str):
                continue

            # 1. Calculate Characters
            total_chars += len(context_text)

            # 2. Calculate Sentences (using your provided logic)
            sentences = nltk.sent_tokenize(context_text)
            total_sentences += len(sentences)

            # 3. Calculate Words (using your provided logic)
            words = re.findall(r'\b\w+\b', context_text, flags=re.UNICODE)
            total_words += len(words)

            valid_items_count += 1

        # Calculate means
        if valid_items_count > 0:
            mean_chars = total_chars / valid_items_count
            mean_words = total_words / valid_items_count
            mean_sentences = total_sentences / valid_items_count
        else:
            mean_chars = mean_words = mean_sentences = 0

        # Store the calculated metrics
        if folder_name not in stats:
            stats[folder_name] = {}
            
        stats[folder_name][model_name] = {
            "mean_characters": mean_chars,
            "mean_words": mean_words,
            "mean_sentences": mean_sentences,
            "items_analyzed": valid_items_count
        }

    return stats

def print_report(stats):
    """Formats and prints the aggregated statistics."""
    if not stats:
        return
        
    print("="*50)
    print("CONTEXT FIELD METRICS REPORT")
    print("="*50)
    
    # Sort folders (results_v1, results_v2, etc.)
    for folder_name in sorted(stats.keys()):
        print(f"\n📂 {folder_name}")
        print("-" * 30)
        
        # Sort models alphabetically (latxa, llama, etc.)
        for model_name in sorted(stats[folder_name].keys()):
            metrics = stats[folder_name][model_name]
            
            print(f"  📄 Model: {model_name}")
            print(f"     • Mean Characters: {metrics['mean_characters']:.2f}")
            print(f"     • Mean Words:      {metrics['mean_words']:.2f}")
            print(f"     • Mean Sentences:  {metrics['mean_sentences']:.2f}")
            print(f"     • Analyzed Items:  {metrics['items_analyzed']}")
            print()

if __name__ == "__main__":
    # Ensure this script is run from the parent directory containing the 'results_*' folders
    # or replace "." with your specific absolute path.
    results_stats = analyze_context_metrics(".")
    print_report(results_stats)