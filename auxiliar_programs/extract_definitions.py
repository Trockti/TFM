import fitz  # PyMuPDF
import ollama
import json
from pydantic import BaseModel
from typing import Dict

class DictionaryEntries(BaseModel):
    entries: Dict[str, str]

def extract_dictionary_page(text_content, model_name='llama3:8b'):

    response = ollama.chat(
        model=model_name,
       messages=[
            {
            'role': 'system',
            'content': 'You are a linguistic data extractor specialized in dictionaries. Your objective is to extract English terms and their main definitions in Spanish.'
        },
        {
            'role': 'user',
            'content': f"""Extract the information from the following text. 
            Strict rules:
            - Key: The entry word or phrase in the original language, not the adaptation. For example, in nani "['na.ni] f. (<ing. nanny)", put nanny, not nani.
            - Value: Only the main definition.
            - Ignore: Phonetic transcriptions in brackets [], examples starting with "Ej.:", and geography or society tags (Geo.:, Soc.:).
            
            Text:
            {text_content}"""
        }
        ],
        format=DictionaryEntries.model_json_schema(), 
        options={'temperature': 0} 
    )
    
    try:
        data = json.loads(response['message']['content'])
        return data.get('entries', {})
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {}

def process_pdf(pdf_path, model_name='llama3:8b'):
    doc = fitz.open(pdf_path)
    full_dictionary = {}
        
    for page_num in range(len(doc)):
        if page_num >= 36 and page_num <= 93:
            print(f"Processing page {page_num + 1}/{len(doc)}...")
            page_text = doc[page_num].get_text()
            page_data = extract_dictionary_page(page_text, model_name)
            full_dictionary.update(page_data)

    return full_dictionary

if __name__ == "__main__":
    result = process_pdf("../data/diccionario_anglicismos.pdf")
    with open("../data/extracted_definitions_cervantes_original.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)