import os
import json

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')      
nltk.download('wordnet')    
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger_eng') 


def combine_dics():
    dirName = "../data"
    definitions = {}
    priority_file = "cervantes_best.json" 
    for file in os.listdir(dirName):
        # To use a specific file, just write it's name in the line below
        if file.endswith("best.json") and file != priority_file:
            print(f"Processing {file}...")
            result = os.path.join(dirName, file)
            with open(result, "r", encoding="utf-8") as f:
                data = json.load(f)
                lemmatizer = WordNetLemmatizer()
                definitions.update({lemmatizer.lemmatize(k.lower()): v for k, v in data.items()})
    if os.path.exists(os.path.join(dirName, priority_file)):
        print(f"Processing {priority_file} with priority...")
        with open(os.path.join(dirName, priority_file), "r", encoding="utf-8") as f:
            data = json.load(f)
            lemmatizer = WordNetLemmatizer()
            definitions.update({lemmatizer.lemmatize(k.lower()): v for k, v in data.items()})
    with open("../data/definitions_v2.json", "w", encoding="utf-8") as f:
        json.dump(definitions, f, indent=2, ensure_ascii=False)


import json

def compare_json_keys(file1_path, file2_path, output_path, different = False):
    # Load the first JSON file
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
        
    # Load the second JSON file
    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
        
    # Get the top-level keys as sets for easy comparison
    keys1 = {k.lower() for k in data1.keys()}
    keys2 = {k.lower() for k in data2.keys()}
    
    if different:
        # Find keys present in one but not the other
        only_in_file1 = list(keys1 - keys2)
        only_in_file2 = list(keys2 - keys1)
        only_in_file1.sort()
        only_in_file2.sort()

        # Prepare the output dictionary
        result = {
            "keys_only_in_file1": only_in_file1,
            "keys_only_in_file2": only_in_file2
        }
    else:
        # Find keys present in both
        common_keys = list(keys1.intersection(keys2))
        common_keys.sort()
        
        # Prepare the output dictionary
        result = {
            "keys_in_both": common_keys
        }
        
        # Save the result to a new JSON file
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(result, out_file, indent=4, ensure_ascii=False)
            
        print(f"Comparison complete. Results saved to {output_path}")



if __name__ == "__main__":
    combine_dics()
    # compare_json_keys('../data/extracted_definitions_extranjetrismos.json', '../data/extracted_definitions_cervantes.json', '../data/key_differences.json')