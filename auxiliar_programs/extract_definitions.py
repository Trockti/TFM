import fitz  # PyMuPDF
import ollama
import json
from pydantic import BaseModel
from typing import Dict

# 1. Define the structure we want using Pydantic
class DictionaryEntries(BaseModel):
    entries: Dict[str, str]

def extract_dictionary_page(text_content):
    # We use the 'format' parameter to enforce JSON output
    response = ollama.chat(
        model='llama3:8b',
       messages=[
            {
                'role': 'system',
                'content': 'Eres un extractor de datos lingüísticos especializado en diccionarios. Tu objetivo es extraer términos en inglés y sus definiciones principales en español.'
            },
            {
                'role': 'user',
                'content': f"""Extrae la información del siguiente texto. 
                Reglas estrictas:
                - Clave (Key): La palabra o frase de entrada (anglicismo).
                - Valor (Value): Solo la definición principal.
                - Ignora: Transcripciones fonéticas entre corchetes [], ejemplos que empiecen con "Ej.:", y marcas de geografía o sociedad (Geo.:, Soc.:).
                
                Texto:
                {text_content}"""
            }
        ],
        format=DictionaryEntries.model_json_schema(), 
        options={'temperature': 0} 
    )
    
    # Parse the structured response
    try:
        data = json.loads(response['message']['content'])
        return data.get('entries', {})
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {}

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_dictionary = {}
        
    for page_num in range(len(doc)):
        if page_num >= 1:  
            print(f"Processing page {page_num + 1}/{len(doc)}...")
            page_text = doc[page_num].get_text()
            page_data = extract_dictionary_page(page_text)
            full_dictionary.update(page_data)

    return full_dictionary

# Execution
result = process_pdf("../data/diccionario-anglicismos-extranjerismos-2021-1.pdf")

with open("../data/extracted_definitions_extranjetrismos.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)