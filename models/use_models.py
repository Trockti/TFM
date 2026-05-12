#!/usr/bin/env python3
"""
Program to run prompts with multiple Hugging Face language models using System/User roles.

Models:
1. meta-llama/Llama-3.2-3B-Instruct - Multilingual Llama model
2. mistralai/Ministral-3-8B-Instruct-2512 - Ministral model
3. HiTZ/Latxa-Qwen3-VL-8B-Instruct - Spanish/Catalan/Basque specialized model
4. IIC/RigoChat-7b-v2 - Spanish language model
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import difflib
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Mistral3ForConditionalGeneration, FineGrainedFP8Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
     "llama": {
        "name": "meta-llama/Llama-3.2-3B-Instruct", 
        "description": "Multilingual Llama 3.2 3B Instruct model",
        "language": "Multilingual",
        "type": "causal_lm",
    },
    "ministral": {
        "name": "mistralai/Ministral-3-8B-Instruct-2512",
        "description": "Ministral model",
        "language": "Multilingual",
        "type": "ministral",
    },
    "latxa": {
        "name": "HiTZ/Latxa-Qwen3-VL-8B-Instruct",
        "description": "Qwen fine-tune for Spanish, Catalan, and Basque",
        "language": "Spanish/Catalan/Basque",
        "type": "vlm",
    },
    "rigochat": {
        "name": "IIC/RigoChat-7b-v2",
        "description": "Spanish language model",
        "language": "Spanish",
        "type": "causal_lm",
    },
}

def normalizar_termino(termino):
    """
    Convierte a minúsculas y elimina espacios y guiones para comparar.
    Ej: 'start-up', 'start up' y 'startup' se convertirán todos en 'startup'.
    """
    return re.sub(r'[\s\-]', '', termino.lower())

def buscar_en_dataset(palabra, ruta_json="../data/normalized_definitions.json"):
    """
    Busca la palabra en el archivo JSON.
    Asume que el JSON tiene la estructura: {"termino": "definicion", ...}
    """
    if not os.path.exists(ruta_json):
        print(f"Advertencia: No se encontró el archivo {ruta_json}")
        return None

    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            datos = json.load(f)
    except json.JSONDecodeError:
        print("Error al leer el JSON. Asegúrate de que el formato sea correcto.")
        return None

    palabra_norm = normalizar_termino(palabra)

    # 1. Búsqueda por normalización (ideal para guiones y espacios)
    for termino_dataset, definicion in datos.items():
        # Check 1: Exact normalized match
        if normalizar_termino(termino_dataset) == palabra_norm:
            return definicion
            
        # Check 2: Match just the abbreviation part before the parenthesis
        # Splits "ai developer (artificial...)" into "ai developer"
        termino_base = termino_dataset.split(' (')[0].strip() 
        if normalizar_termino(termino_base) == palabra_norm:
            return definicion
            
    # 2. Búsqueda difusa (fuzzy search) para errores ortográficos leves
    try:
        with open("../data/definitions_v2.json", 'r', encoding='utf-8') as f:
            datos = json.load(f)
    except json.JSONDecodeError:
        print("Error al leer el JSON. Asegúrate de que el formato sea correcto.")
        return None
        
    terminos_disponibles = list(datos.keys())
    # cutoff=0.9 significa que debe haber al menos un 90% de similitud
    coincidencias = difflib.get_close_matches(palabra.lower(), terminos_disponibles, n=1, cutoff=0.7)
    
    if coincidencias:
        termino_encontrado = coincidencias[0]
        return datos[termino_encontrado]

    return None


class ModelManager:
    """Manages loading and inference with language models."""

    def __init__(self, device: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the ModelManager.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            cache_dir: Directory to cache models
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.cache_dir = cache_dir
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        logger.info("Using device: %s", self.device)

    def load_model(self, model_key: str, load_in_8bit: bool = False) -> bool:
        """
        Load a model from the MODELS dictionary.

        Args:
            model_key: Key identifying the model (e.g., 'latxa')
            load_in_8bit: Whether to load causal LMs in 8-bit precision

        Returns:
            True if successful, False otherwise
        """
        if model_key not in MODELS:
            logger.error("Unknown model key: %s", model_key)
            return False

        if model_key in self.loaded_models:
            logger.info("Model %s already loaded", model_key)
            return True

        model_name = MODELS[model_key]["name"]
        model_type = MODELS[model_key].get("type", "causal_lm")
        logger.info("Loading model: %s (%s)", model_name, MODELS[model_key]["description"])

        try:
            if model_type == "vlm":
                pipe_kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "trust_remote_code": True,
                }
                if self.cache_dir:
                    pipe_kwargs["cache_dir"] = self.cache_dir
                if self.device == "cuda":
                    pipe_kwargs["device_map"] = "auto"
                    pipe_kwargs["dtype"] = torch.float16

                vlm_pipe = pipeline("image-text-to-text", **pipe_kwargs)
                self.loaded_models[model_key] = {
                    "type": "vlm",
                    "pipeline": vlm_pipe,
                }
            elif model_type == "ministral":
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto",
                    quantization_config=FineGrainedFP8Config(dequantize=True),
                    cache_dir=self.cache_dir,
                )
                self.loaded_models[model_key] = {
                    "type": "causal_lm",
                    "model": model,
                    "tokenizer": tokenizer,
                }
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model_kwargs: Dict[str, Any] = {
                    "cache_dir": self.cache_dir,
                    "device_map": "auto",
                    "trust_remote_code": True,
                }
                if self.device == "cuda":
                    model_kwargs["dtype"] = torch.float16
                    if load_in_8bit:
                        model_kwargs["load_in_8bit"] = True

                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                self.loaded_models[model_key] = {
                    "type": "causal_lm",
                    "model": model,
                    "tokenizer": tokenizer,
                }

            logger.info("Successfully loaded %s", model_key)
            return True
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_key, str(exc))
            return False

    def generate(
        self,
        model_key: str,
        messages: List[Dict[str, str]], # Changed from prompt: str
        max_length: int = 256,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Optional[List[str]]:
        """
        Generate text using the specified model.

        Returns:
            List of generated texts, or None if generation fails
        """
        if model_key not in self.loaded_models:
            logger.error("Model %s not loaded", model_key)
            return None

        model_obj = self.loaded_models[model_key]
        model_type = model_obj.get("type", "causal_lm")

        try:
            if model_type == "vlm":
                # Adapt standard messages to VLM text blocks
                vlm_messages = []
                for msg in messages:
                    vlm_messages.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                
                outputs = model_obj["pipeline"](
                    text=vlm_messages,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )
                return self._extract_vlm_text(outputs)

            model = model_obj["model"]
            tokenizer = model_obj["tokenizer"]

            # MAGIC HAPPENS HERE: Apply chat template transforms system/user dicts into model-specific tokens
            prompt_str = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = tokenizer(
                prompt_str,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_length, 
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # We slice the output to avoid returning the prompt text along with the generated response
            input_length = inputs.input_ids.shape[1]
            generated_ids = output_ids[:, input_length:]
            
            return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
        except Exception as exc:
            logger.error("Generation failed for %s: %s", model_key, str(exc))
            return None

    @staticmethod
    def _extract_vlm_text(outputs: Any) -> Optional[List[str]]:
        """Normalize common image-text-to-text output formats to plain text."""
        if not isinstance(outputs, list):
            return [str(outputs)]

        extracted: List[str] = []
        for item in outputs:
            if not isinstance(item, dict):
                extracted.append(str(item))
                continue

            generated = item.get("generated_text", "")
            if isinstance(generated, str):
                extracted.append(generated)
            elif isinstance(generated, list):
                assistant_messages: List[str] = []
                for msg in generated:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            assistant_messages.append(content)
                        else:
                            assistant_messages.append(str(content))
                extracted.append("\n".join([m for m in assistant_messages if m]))
            else:
                extracted.append(str(generated))

        return extracted if extracted else None

    def unload_model(self, model_key: str) -> None:
        """Unload a model to free memory."""
        import gc
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded model: %s", model_key)

    def list_loaded_models(self) -> List[str]:
        """Return list of loaded model keys."""
        return list(self.loaded_models.keys())


def generate_with_model(
    prompts: List[List[Dict[str, str]]], # Changed typing to expect standard message lists
    model_key: str,
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    load_in_8bit: bool = False,
) -> List[Optional[str]]: # Now returns an ordered list instead of dictionary
    """Generate text using a specified model and prompts."""
    manager = ModelManager(device=device, cache_dir=cache_dir)
    
    results: List[Optional[str]] = []

    logger.info("\n--- Loading model: %s ---", model_key)
    if not manager.load_model(model_key, load_in_8bit=load_in_8bit):
        logger.warning("Could not load model: %s", model_key)
        return [None] * len(prompts)

    for i, messages in enumerate(prompts, 1):
        logger.info("Processing prompt %d/%d with %s...", i, len(prompts), model_key)
        texts = manager.generate(
            model_key,
            messages,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )
        if texts:
            results.append(texts[0])
            logger.info("    Generated %d characters", len(texts[0]))
        else:
            results.append(None)
            logger.warning("    Generation failed")
            
    # Unload model to free memory for the next one
    manager.unload_model(model_key)

    return results


def main() -> None:
    """Main entry point - modify prompts and models here."""
    
    # 1. Load your glossary dataset 
    input_file = "../data/transformed_terms.json" 
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please ensure your dataset is available.")
        return

    # 2. Define your prompt templates AS MESSAGE LISTS
    prompt_templates = [
            # Un único prompt unificado que incluye rol, instrucciones, contexto, definición y ejemplos few-shot en un solo bloque
            [
                {"role": "system", "content": """Eres un especialista en lenguaje ciudadano, accesibilidad cognitiva y Lectura Fácil. Tu tarea consiste en traducir y explicar anglicismos complejos al español de manera que cualquier persona pueda entenderlos sin dificultad.

    Instrucciones:
    1. Redacta una definición breve, sencilla y directa.
    2. Utiliza frases cortas y vocabulario cotidiano.
    3. Basa tu explicación en cómo se utiliza la palabra en el 'Contexto' proporcionado.
    4. Utiliza la 'Definición formal' como referencia de precisión (si está disponible), pero adáptala para que sea fácil de entender.
    5. Si es útil, añade un ejemplo breve o explica cómo se pronuncia.

    Aquí tienes algunos ejemplos de cómo debes realizar la tarea:

    Ejemplo 1:
    - Término: App
    - Contexto: "Los primeros minutos en la app fueron un bombardeo de mensajes..."
    - Definición formal: Programa preparado para una utilización específica, como el pago de nóminas, el tratamiento de textos, et.
    - Tu respuesta: Son programas informáticos que se instalan en el teléfono móvil, ordenador o tablet para hacer tareas concretas, como jugar a juegos, escribir mensajes o trabajar con documentos. Por ejemplo, el WhatsApp es una aplicación. También se llaman Apps. Pueden ser de pago o gratis.

    Ejemplo 2:
    - Término: Phishing
    - Contexto: "En este escenario, los ataques se volverán cada vez más complejos, y se combinarán contenidos como spam y phishing."
    - Definición formal: El phishing es una técnica que consiste en el envío de un correo electrónico por parte de un ciberdelincuente a un usuario simulando ser una entidad legítima (red social, banco, institución pública, etc.) con el objetivo de robarle información privada, realizarle un cargo económico o infectar el dispositivo. Para ello, adjuntan archivos infectados o enlaces a páginas fraudulentas en el correo electrónico.
    - Tu respuesta: Es un tipo de engaño que consiste en el envío de emails falsos para que entres en una página web falsa y escribas tus datos.

    Ejemplo 3:
    - Término: Router
    - Contexto: "AlphaShield es un cortafuegos que se instala entre el router o módem del usuario y el ordenador..."
    - Definición formal: Dispositivo que distribuye el flujo de paquetes de información entre redes de la manera más eficaz.
    - Tu respuesta: Aparato que sirve para que uno o varios dispositivos como, por ejemplo, ordenadores, teléfonos, tabletas y televisores tengan conexión a Internet. Es una palabra inglesa y se pronuncia ruter."""},
                
                {"role": "user", "content": """Ahora es tu turno. Adapta el siguiente anglicismo siguiendo exactamente el mismo estilo de los ejemplos anteriores:

    - Término: '{term}'
    - Contexto: '{context}'
    - Definición formal: '{definition}'
    - Tu respuesta:"""}
            ]
        ]

    model_keys = list(MODELS.keys())
    
    # 3. Iterate over each prompt template (generating folder v1, v2...)
    for idx, prompt_template in enumerate(prompt_templates, start=1):
        folder_name = f"results_v{idx+3}"  # Start from v4 to avoid confusion with previous versions
        folder_path = Path(folder_name)
        folder_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n" + "=" * 80)
        print(f"Processing Prompt V{idx} -> Directory: {folder_name}")
        print("=" * 80)
        
        # 4. Iterate over each model
        for model_key in model_keys:
            print(f"Running generation for model: {model_key}...")
            
            # Prepare all formatted prompts for batch generation
            formatted_prompts = []
            for field_name, item in dataset.items():
                term = item.get("term", "")
                
                # Format each message block individually
                messages = []
                for msg_block in prompt_template:
                    messages.append({
                        "role": msg_block["role"],
                        "content": msg_block["content"].format(
                            term=term,
                            context=item.get("context", ""),
                            original=item.get("original", ""),
                            definition=buscar_en_dataset(term) or ""
                        )
                    })
                    
                formatted_prompts.append(messages)
            
            # Run the model on the full batch of message prompts
            model_results_list = generate_with_model(
                prompts=formatted_prompts,
                model_key=model_key,
                max_length=1024,
                temperature=0.7,
                top_p=0.9,
                load_in_8bit=False,
            )
            
            # 5. Re-assemble the output to match the requested structure
            model_output_data = {}
            
            # Use zip to perfectly pair the original dictionary items with the generated results list
            for (field_name, item), generated_text in zip(dataset.items(), model_results_list):
                
                model_output_data[field_name] = {
                    "id_term": item.get("id_term", ""),
                    "term": item.get("term", ""),
                    "original": item.get("original", ""),
                    "adapted": item.get("adapted", ""),
                    "id_definition": item.get("id_definition", ""),
                    "id_adaptation": item.get("id_adaptation", ""),
                    "context": item.get("context", ""),
                    "simplified": generated_text if generated_text else "(Generation failed)"
                }
            
            # 6. Save the model's JSON file inside the specific version folder
            output_file = folder_path / f"{model_key}.json"
            with output_file.open("w", encoding="utf-8") as file_handle:
                json.dump(model_output_data, file_handle, indent=4, ensure_ascii=False)
            
            print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()