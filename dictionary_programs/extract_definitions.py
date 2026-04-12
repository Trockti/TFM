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
        'content': 'You are a linguistic data extractor specialized in dictionaries. Your objective is to extract English terms (anglicisms) and their main definitions in Spanish, outputting them strictly as a JSON object. You must not insert Spanish words as keys. For definitions introduced by filler phrases like "Anglicismo por", "Anglicismo por el", or "Anglicismo por de la", strip those introductory words completely and save only the core definition.'
    },
    {
        'role': 'user',
        'content': f"""Extract the information from the following text and format it as a single JSON object. 

    Strict rules:
    - Key: The exact entry word or phrase (anglicism), including any parentheses.
    - Value: Only the core main definition.
    - Ignore: Phonetic transcriptions in brackets [], examples starting with "Ej.:", and geography or society tags (Geo.:, Soc.:).
    - Output ONLY valid JSON. Do not include markdown blocks, conversational text, or explanations.

    Examples:
    Text: "EBIT (EARNINGS BEFORE INTERESTS AND TAXES). Siglas inglesas que significan beneficio (empresarial) antes de intereses e impuestos."
    Output: {{"EBIT (EARNINGS BEFORE INTERESTS AND TAXES)": "beneficio (empresarial) antes de intereses e impuestos"}}

    Text: "A2C (ADMINISTRATION-TO-CONSUMER). Anglicismo por de la Administración al administrado, de la Administración al consumidor. Ej.: El modelo A2C."
    Output: {{"A2C (ADMINISTRATION-TO-CONSUMER)": "Administración al administrado, de la Administración al consumidor."}}

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
        if page_num >= 1:
            print(f"Processing page {page_num + 1}/{len(doc)}...")
            page_text = doc[page_num].get_text()
            page_data = extract_dictionary_page(page_text, model_name)
            full_dictionary.update(page_data)

    return full_dictionary

if __name__ == "__main__":
    result = process_pdf("../data/diccionario-anglicismos-extranjerismos-2021.pdf")
    with open("../data/extracted_definitions_damaso_1_v5.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)