import json
import re
from pathlib import Path
import pandas as pd
import nltk
from langdetect import detect, LangDetectException
import pyphen

# Download the sentence tokenizer model for NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def no_double_negation_rule(text):
    """Detects simple double negations."""
    if not isinstance(text, str):
        return True
    text = text.lower()
    patterns = [
        r"\bno\b.*\bnunca\b",
        r"\bno\b.*\bnada\b",
        r"\bno\b.*\bnadie\b",
        r"\bno\b.*\bningún\b",
        r"\bno\b.*\bninguna\b",
        r"\bno\b.*\bninguno\b"
    ]
    return not any(re.search(pattern, text) for pattern in patterns)

def active_voice_proxy_rule(text):
    """Proxy to detect passive voice."""
    if not isinstance(text, str):
        return True
    text = text.lower()
    passive_patterns = [
        r"\bes\b\s+\w+ado\b",
        r"\bes\b\s+\w+ido\b",
        r"\bfue\b\s+\w+ado\b",
        r"\bfue\b\s+\w+ido\b",
        r"\bson\b\s+\w+ados\b",
        r"\bson\b\s+\w+idos\b",
        r"\bse utiliza\b",
        r"\bse usa\b"
    ]
    return not any(re.search(pattern, text) for pattern in passive_patterns)

def category_function_pattern_rule(text):
    """Checks if it starts with the typical structure 'category + function'."""
    if not isinstance(text, str):
        return False
    text = text.strip().lower()
    patterns = [
        "aparato que", "persona que", "personas que", "actividad que",
        "tecnología que", "sistema que", "servicio que", "forma de",
        "conjunto de", "documento que", "lugar donde", "acción de",
        "deporte que", "técnica que"
    ]
    return any(text.startswith(pattern) for pattern in patterns)

def pronunciation_presence(text):
    """Detects if pronunciation is included."""
    if not isinstance(text, str):
        return False
    text = text.lower()
    patterns = ["se pronuncia", "pronunciación", "se lee"]
    return any(pattern in text for pattern in patterns)

def example_presence(text):
    """Detects if an example is included."""
    if not isinstance(text, str):
        return False
    text = text.lower()
    patterns = ["por ejemplo", "ejemplo:", "como cuando", "como, por ejemplo"]
    return any(pattern in text for pattern in patterns)

def analyze_text(text, term=""):
    """Analyzes a text block and returns all specified metrics and rules."""
    # 1. Base Checks & Defaults
    if not text or not isinstance(text, str):
        return {
            "Language": "unknown", "Words": 0, "Syllables": 0, "Phrases": 0,
            "Avg_Sentence_Length": 0, "Has_Pronunciation": False, "Has_Example": False,
            "Rule_Max_25_Words": False, "Rule_Max_2_Sentences": False,
            "Rule_Short_Sentences": False, "Rule_No_Double_Neg": True,
            "Rule_No_Self_Def": True, "Rule_Active_Voice": True,
            "Rule_Cat_Func_Pattern": False, "Prompt_Compliance_Score": 0.0
        }

    # 2. Detect Language
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"

    # 3. Structural Metrics
    sentences = nltk.sent_tokenize(text)
    num_phrases = len(sentences)
    
    words = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    num_words = len(words)
    
    avg_sentence_length = (num_words / num_phrases) if num_phrases > 0 else 0

    # 4. Count Syllables
    dic_lang = lang if lang in pyphen.LANGUAGES else 'es'
    try:
        dic = pyphen.Pyphen(lang=dic_lang)
    except KeyError:
        dic = pyphen.Pyphen(lang='es')
    num_syllables = sum([len(dic.positions(w)) + 1 for w in words])

    # 5. Rule Evaluations
    rule_max_25_words = num_words <= 25
    rule_max_2_sentences = num_phrases <= 2
    rule_short_sentences = avg_sentence_length <= 15
    rule_no_double_neg = no_double_negation_rule(text)
    
    # Self-definition rule relies on the term being provided
    if term and isinstance(term, str):
        rule_no_self_def = term.lower() not in text.lower()
    else:
        rule_no_self_def = True
        
    rule_active_voice = active_voice_proxy_rule(text)
    rule_cat_func_pattern = category_function_pattern_rule(text)

    # 6. Additional Signals
    has_pronunciation = pronunciation_presence(text)
    has_example = example_presence(text)

    # 7. Prompt Compliance Score Calculation
    rules_dict = {
        "max_25_words": rule_max_25_words,
        "max_2_sentences": rule_max_2_sentences,
        "short_sentences": rule_short_sentences,
        "no_double_negation": rule_no_double_neg,
        "no_self_definition": rule_no_self_def,
        "active_voice_proxy": rule_active_voice,
        "category_function_pattern": rule_cat_func_pattern
    }
    compliance_score = sum(rules_dict.values()) / len(rules_dict)

    return {
        "Language": lang,
        "Words": num_words,
        "Syllables": num_syllables,
        "Phrases (Sentences)": num_phrases,
        "Avg Sentence Length": round(avg_sentence_length, 2),
        "Has Pronunciation": has_pronunciation,
        "Has Example": has_example,
        "Compliance: Max 25 Words": rule_max_25_words,
        "Compliance: Max 2 Sentences": rule_max_2_sentences,
        "Compliance: Short Sentences (<15 words/sent)": rule_short_sentences,
        "Compliance: No Double Negation": rule_no_double_neg,
        "Compliance: No Self-Definition": rule_no_self_def,
        "Compliance: Active Voice Proxy": rule_active_voice,
        "Compliance: Category+Function Pattern": rule_cat_func_pattern,
        "Prompt Compliance Score": round(compliance_score, 3)
    }

def main(base_directory="../models"):
    results = []
    base_path = Path(base_directory)
    
    for folder_path in base_path.glob("results_*"):
        if folder_path.is_dir():
            folder_name = folder_path.name
            
            for json_file in folder_path.glob("*.json"):
                model_name = json_file.stem
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error reading JSON in {json_file}. Skipping.")
                        continue
                
                for entry_id, entry_data in data.items():
                    simplified_text = entry_data.get("simplified", "")
                    # Try to extract the target term to test the "self-definition" rule properly
                    term = entry_data.get("term", "")
                    
                    # Get all metrics as a dictionary
                    metrics = analyze_text(simplified_text, term=term)
                    
                    # Store the results, spreading the returned metrics dictionary
                    row_data = {
                        "Folder": folder_name,
                        "Model": model_name,
                        "ID": entry_id,
                        **metrics
                    }
                    results.append(row_data)

    df = pd.DataFrame(results)
    
    if not df.empty:
        output_file_csv = "text_analysis_results.csv"
        df.to_csv(output_file_csv, index=False, encoding='utf-8-sig')
        
        output_file_json = "text_analysis_results.json"
        df.to_json(output_file_json, orient="records", indent=4, force_ascii=False)
        
        print(f"Analysis complete! Processed {len(df)} texts.")
        print(f"Results saved to: {output_file_csv} and {output_file_json}")
        
        print("\nPreview of results:")
        print(df.head())
    else:
        print("No valid data found to process.")

if __name__ == "__main__":
    main("../models")