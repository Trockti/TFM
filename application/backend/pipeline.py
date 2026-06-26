#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments
)
import requests
import string 

# from RAE import buscar_palabra_rae
import spacy


# Función para cargar el modelo entrenado y su tokenizer
def load_model(model_path="./borrowing_classifier_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Configurar argumentos de entrenamiento básicos para el Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
        push_to_hub=False,
    )
    
    # Crear el trainer para predicciones
    trainer = Trainer(
        model=model,
        args=training_args,
    )
    
    return tokenizer, trainer

# Cargar las etiquetas del modelo
def load_labels(model_path="./borrowing_classifier_model"):
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return model.config.id2label, model.config.label2id

# Función para predecir entidades en un texto con anotaciones de POS
def predict_borrowings_with_pos(text, trainer, tokenizer):
    # Get model device
    device = trainer.model.device

    # Tokenize the input text
    tokens = text.split()  # Simple tokenization
    tokenized_inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)

    # Store word_ids before moving to device
    word_ids = tokenized_inputs.word_ids(batch_index=0)

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = trainer.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Process predictions
    predicted_ner_labels = []
    id2label = trainer.model.config.id2label
    
    previous_word_idx = None
    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue  # Skip special tokens

        # Only take the first subword of each word
        if word_id != previous_word_idx:
            predicted_idx = predictions[0, i].item()
            predicted_ner_label = id2label[predicted_idx]
            predicted_ner_labels.append(predicted_ner_label)
            previous_word_idx = word_id

    # Pair tokens with their predicted NER labels
    result = list(zip(tokens, predicted_ner_labels))
    return result

def obtain_sentences(file_url):
    sentences = []
    response = requests.get(file_url)
    content = response.text.splitlines()

    sentence = []
    for line in content:
        line = line.strip()
        if line == "":  # End of sentence
            if sentence:
                sentences.append(sentence)
                sentence = []
        else:
            parts = line.split(" ")  # Assuming words and labels are separated by a tab
            sentence.append(parts[0])
    # Add last sentence if file doesn't end with newline
    if sentence:
        sentences.append(sentence)
    return sentences

def create_sentence(words):
    """
    Create a grammatically correct sentence from a list of words.

    Args:
        words (list): A list of strings representing words

    Returns:
        str: A properly formatted sentence
    """
    if not words:
        return ""

    # Join the words with spaces
    sentence = " ".join(words)

    return sentence

# Función para procesar un lote de oraciones y guardar resultados
def process_sentence_batch(sentences, output_file_path="prediction_results.txt", conll_path='conll_predictions.conll'):
    # Cargar el modelo
    tokenizer, trainer = load_model()
    
    # Abrir archivos para escribir resultados
    with open(output_file_path, 'w', encoding='utf-8') as output_file, open(conll_path, 'w', encoding='utf-8') as conll:
        for sentence_words in sentences:
            phrase = create_sentence(sentence_words)
            predictions = predict_borrowings_with_pos(phrase, trainer, tokenizer)
            
            # Escribir resultados en formato CoNLL
            for word, ner_label in predictions:
                conll.write(f"{word}\tPOS\t{ner_label}\n")
            conll.write("\n")  # Separar oraciones

            # Escribir resultados en formato legible
            output_file.write(f"Sentence: {phrase}\n")
            output_file.write(f"Predictions: {predictions}\n\n")
    
    print(f"Resultados guardados en {output_file_path} y {conll_path}")

def predecir_anglicismo(load_model, predict_borrowings_with_pos, text):
    text = text.replace("'", "").replace('"', "")
    nlp = spacy.load('es_core_news_md')
    # nlp = spacy.load('en_core_web_trf')

    tokenizer, trainer = load_model()
    predictions = predict_borrowings_with_pos(text, trainer, tokenizer)
    print("Predictions:", predictions)

    # Identificar tokens que necesitan verificación (no etiquetados como "O")
    tokens_to_check = []
    indices_to_check = []
    # tokens_O = []
    
    
    for i, (token, label) in enumerate(predictions):
        # print(f"{token}: {label}")
        if label != "O":
            # Eliminar signos de puntuación del token antes de procesarlo
            predictions[i] = (token.translate(str.maketrans('', '', ",.?:)!")), label)
            tokens_to_check.append(predictions[i][0])
            indices_to_check.append(i)
        # else:
        #     tokens_O.append(token)
    
    # Procesar todos los tokens de una vez con spaCy
    docs = list(nlp.pipe(tokens_to_check))
    
    # Verificar cada token
    for j, (token, doc) in enumerate(zip(tokens_to_check, docs)):
        i = indices_to_check[j]
        # print(doc[0].lemma_)
        # rae_result = buscar_palabra_rae(doc[0].lemma_)
        
        # # # Manejar el caso donde rae_result es None
        # if rae_result is None:
        #     print(f"Warning: buscar_palabra_rae returned None for token '{token}'")
        #     # Mantener la etiqueta original
        #     # if doc[0].pos_ == "PROPN" or doc[0].is_digit or doc[0].like_email:
        #     # # Nombre propio, dígito o email - no es préstamo
        #     #     predictions[i] = (token, "O")
        #     continue
            
        # if rae_result["estado"] == 1:
        #     # Palabra encontrada en RAE - no es préstamo
        #     predictions[i] = (token, "O")
        if doc[0].is_digit or doc[0].like_email:
        #     # Nombre propio, dígito o email - no es préstamo
        #     # print(doc[0].pos_)
        #     # print(doc[0].tag_)
        #     # print(doc[0].ent_type_)
            predictions[i] = (token, "O")
        # print(doc[0].pos_, spacy.explain(doc[0].pos_))
    # for j, token in enumerate(tokens_O):
    #     if buscar_palabra_rae(token) is None:
    #         # Palabra encontrada en RAE - no es préstamo
    #         predictions[j] = (token, "B-ENG")


    print("Predictions new:", predictions)
    
    return predictions

if __name__ == "__main__":
    text = '"""Burpees"" para perder kilos sin salir de casa'
    model_path = "./models/borrowing_classifier_model_extended_v2"
    predecir_anglicismo(lambda: load_model(model_path), predict_borrowings_with_pos, text)