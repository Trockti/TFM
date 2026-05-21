import json
import re
from pathlib import Path
import pandas as pd
import nltk
from langdetect import detect, LangDetectException
import pyphen

# Download the sentence tokenizer model for NLTK
nltk.download('punkt', quiet=True)

def analyze_text(text):
    """Analyzes a text block and returns word count, syllables, phrases, and language."""
    # Handle empty or missing text
    if not text or not isinstance(text, str):
        return 0, 0, 0, "unknown"

    # 1. Detect Language
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"

    # 2. Count Phrases (Sentences)
    # NLTK punkt handles punctuation and abbreviations better than simple splitters
    sentences = nltk.sent_tokenize(text)
    num_phrases = len(sentences)

    # 3. Count Words
    # Finds all sequences of alphanumeric characters, ignoring punctuation/markdown
    words = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    num_words = len(words)

    # 4. Count Syllables
    # Initialize Pyphen dictionary for the detected language (fallback to Spanish 'es')
    dic_lang = lang if lang in pyphen.LANGUAGES else 'es'
    try:
        dic = pyphen.Pyphen(lang=dic_lang)
    except KeyError:
        dic = pyphen.Pyphen(lang='es')

    # A word's syllables = number of hyphenation points + 1
    num_syllables = sum([len(dic.positions(word)) + 1 for word in words])

    return num_words, num_syllables, num_phrases, lang

def main(base_directory="../models"):
    results = []
    base_path = Path(base_directory)
    # Find all folders that match "result_*"
    for folder_path in base_path.glob("results_*"):
        if folder_path.is_dir():
            folder_name = folder_path.name
            
            # Find all JSON files inside the folder
            for json_file in folder_path.glob("*.json"):
                model_name = json_file.stem  # Gets the filename without the .json extension
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error reading JSON in {json_file}. Skipping.")
                        continue
                
                # Iterate through each entry in the JSON
                for entry_id, entry_data in data.items():
                    simplified_text = entry_data.get("simplified", "")
                    
                    # Get metrics
                    words, syllables, phrases, lang = analyze_text(simplified_text)
                    
                    # Store the results
                    results.append({
                        "Folder": folder_name,
                        "Model": model_name,
                        "ID": entry_id,
                        "Language": lang,
                        "Words": words,
                        "Syllables": syllables,
                        "Phrases (Sentences)": phrases
                    })

    # Convert results to a pandas DataFrame for easy viewing and export
    df = pd.DataFrame(results)
    
    if not df.empty:
        # Save to CSV
        output_file_csv = "text_analysis_results.csv"
        df.to_csv(output_file_csv, index=False, encoding='utf-8-sig')
        
        # Save to JSON
        output_file_json = "text_analysis_results.json"
        # orient="records" creates a list of dictionaries, which is usually the desired JSON format
        df.to_json(output_file_json, orient="records", indent=4, force_ascii=False)
        
        print(f"Analysis complete! Processed {len(df)} texts.")
        print(f"Results saved to: {output_file_csv} and {output_file_json}")
        
        # Print a preview of the first few rows
        print("\nPreview of results:")
        print(df.head())
    else:
        print("No valid data found to process.")

if __name__ == "__main__":
    main("../models")