import json
import logging
from pathlib import Path
from typing import Dict

# Import the ModelManager and MODELS config from your existing file
from use_models import ModelManager, MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_input_data(file_path: str) -> Dict[str, str]:
    """Loads the word:definition JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_simplification(input_file: str):
    """
    Iterates through models and words to create simplified definitions.
    """
    data = load_input_data(input_file)
    manager = ModelManager()

    for model_key, info in MODELS.items():
        model_name_clean = info["name"].split('/')[-1]
        output_filename = f"{model_name_clean}.json"
        
        logger.info(f"--- Processing Model: {model_key} ({info['name']}) ---")
        
        if not manager.load_model(model_key):
            logger.error(f"Skipping {model_key} due to load failure.")
            continue

        results = {}

        for word, definition in data.items():
            prompt = (
                f"Explain the following definition in very simple terms for a child. "
                f"Keep it concise.\n\n"
                f"Word: {word}\n"
                f"Definition: {definition}\n\n"
                f"Simplified:"
            )

            logger.info(f"Simplifying '{word}' with {model_key}...")
            
            # Generate response
            output = manager.generate(
                model_key, 
                prompt, 
                max_length=150, 
                temperature=0.3 # Lower temperature for more focused definitions
            )

            simplified_text = output.strip() if output else "Generation failed"
            
            # Formatting per your requirement
            results[word] = {
                "original": definition,
                "simplified": simplified_text,
                "reference": ""
            }

        # Save model-specific JSON
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_filename}")

        # Unload to free up VRAM for the next model
        manager.unload_model(model_key)

if __name__ == "__main__":
    # Ensure you have a file named 'input_definitions.json' in the directory
    # or change this string to your actual filename.
    input_json_path = ".json"
    
    if Path(input_json_path).exists():
        run_simplification(input_json_path)
    else:
        logger.error(f"Input file {input_json_path} not found.")