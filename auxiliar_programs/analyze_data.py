import os
import json

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')      
nltk.download('wordnet')    
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger_eng') 


def main():
    dirName = "../data"
    definitions = {}
    for file in os.listdir(dirName):
        # To use a specific file, just write it's name in the line below
        if file.endswith("definitions.json"):
            print(f"Processing {file}...")
            result = os.path.join(dirName, file)
            with open(result, "r", encoding="utf-8") as f:
                data = json.load(f)
                lemmatizer = WordNetLemmatizer()
                definitions.update({lemmatizer.lemmatize(k.lower()): v for k, v in data.items()})
    with open("../data/definitions_lemma.json", "w", encoding="utf-8") as f:
        json.dump(definitions, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()