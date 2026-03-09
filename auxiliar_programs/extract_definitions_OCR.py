import fitz  # PyMuPDF
import ollama
import json
from pydantic import BaseModel
from typing import Dict
import pdf2image
from PIL import Image

import pytesseract

def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(pdf_file)


def ocr_core(file):
    text = pytesseract.image_to_string(file, lang='spa')
    return text


class DictionaryEntries(BaseModel):
    entries: Dict[str, str]

def extract_dictionary_page(text_content):

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
    
    try:
        data = json.loads(response['message']['content'])
        return data.get('entries', {})
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {}

def process_pdf(pdf_path):
    full_dictionary = {}
        
    images = pdf_to_img(pdf_path)
    for page_num, img in enumerate(images):
        
        page_text = ocr_core(img)
        print(f"Processing page {page_num + 1}/{len(images)}...")
        page_data = extract_dictionary_page(page_text)
        full_dictionary.update(page_data)

    return full_dictionary

result = process_pdf("../data/Diccionario_de_anglicismos_actuales.pdf")

with open("../data/extracted_definitions_escaners_fotografias.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)