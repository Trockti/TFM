#!/usr/bin/env python3
"""
Program to use prompts with multiple language models from Hugging Face.

Models:
1. meta-llama/Llama-3.2-3B - Multilingual Llama model
2. mistralai/Mixtral-8x7B-v0.1 - Multilingual mixture of experts
3. HiTZ/Latxa-Qwen3-VL-8B-Instruct - Spanish/Catalan/Basque specialized model
4. IIC/RigoChat-7b-v2 - Spanish language model
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "llama": {
        "name": "meta-llama/Llama-3.2-3B",
        "description": "Multilingual Llama 3.2 3B model",
        "language": "Multilingual",
    },
    "mixtral": {
        "name": "mistralai/Mixtral-8x7B-v0.1",
        "description": "Mixtral 8x7B mixture of experts model",
        "language": "Multilingual",
    },
    "latxa": {
        "name": "HiTZ/Latxa-Qwen3-VL-8B-Instruct",
        "description": "Qwen fine-tune for Spanish, Catalan, and Basque",
        "language": "Spanish/Catalan/Basque",
    },
    "rigochat": {
        "name": "IIC/RigoChat-7b-v2",
        "description": "Spanish language model",
        "language": "Spanish",
    },
}


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
        self.loaded_models: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}
        logger.info(f"Using device: {self.device}")

    def load_model(self, model_key: str, load_in_8bit: bool = False) -> bool:
        """
        Load a model from the MODELS dictionary.

        Args:
            model_key: Key identifying the model (e.g., 'llama', 'mixtral')
            load_in_8bit: Whether to load model in 8-bit precision

        Returns:
            True if successful, False otherwise
        """
        if model_key not in MODELS:
            logger.error(f"Unknown model key: {model_key}")
            return False

        if model_key in self.loaded_models:
            logger.info(f"Model {model_key} already loaded")
            return True

        model_name = MODELS[model_key]["name"]
        logger.info(f"Loading model: {model_name} ({MODELS[model_key]['description']})")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with quantization if requested
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "device_map": "auto",
                "trust_remote_code": True,
            }

            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                if load_in_8bit:
                    model_kwargs["load_in_8bit"] = True

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )

            self.loaded_models[model_key] = (model, tokenizer)
            logger.info(f"Successfully loaded {model_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {str(e)}")
            return False

    def generate(
        self,
        model_key: str,
        prompt: str,
        max_length: int = 256,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Optional[List[str]]:
        """
        Generate text using the specified model.

        Args:
            model_key: Key identifying the model
            prompt: Input prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            List of generated texts, or None if generation fails
        """
        if model_key not in self.loaded_models:
            logger.error(f"Model {model_key} not loaded")
            return None

        model, tokenizer = self.loaded_models[model_key]

        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return texts

        except Exception as e:
            logger.error(f"Generation failed for {model_key}: {str(e)}")
            return None

    def unload_model(self, model_key: str) -> None:
        """Unload a model to free memory."""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_key}")

    def list_loaded_models(self) -> List[str]:
        """Return list of loaded models."""
        return list(self.loaded_models.keys())


def generate_with_models(
    prompts: List[str],
    model_keys: List[str] = None,
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    load_in_8bit: bool = False,
    output_file: Optional[Path] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Generate text using specified models and prompts.

    Args:
        prompts: List of text prompts
        model_keys: List of model keys to use (default: all models)
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device to use ('cuda', 'cpu', or None for auto)
        cache_dir: Directory to cache models
        load_in_8bit: Load models in 8-bit precision
        output_file: Path to save results as JSON

    Returns:
        Dictionary mapping prompts to model outputs
    """
    if model_keys is None:
        model_keys = list(MODELS.keys())

    manager = ModelManager(device=device, cache_dir=cache_dir)

    # Load models
    logger.info(f"Loading {len(model_keys)} model(s)...")
    for model_key in model_keys:
        if not manager.load_model(model_key, load_in_8bit=load_in_8bit):
            logger.warning(f"Could not load model: {model_key}")

    if not manager.list_loaded_models():
        logger.error("No models were successfully loaded")
        return {}

    # Generate text for each prompt
    results = {}
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\nProcessing prompt {i}/{len(prompts)}: {prompt[:50]}...")
        results[prompt] = {}

        for model_key in manager.list_loaded_models():
            logger.info(f"  Generating with {model_key}...")
            texts = manager.generate(
                model_key,
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
            )
            if texts:
                results[prompt][model_key] = texts[0]  # Take first generation
                logger.info(f"    Generated {len(texts[0])} characters")
            else:
                results[prompt][model_key] = None
                logger.warning(f"    Generation failed")

    # Save results if output file specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to: {output_file}")

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for prompt, model_results in results.items():
        print(f"\nPrompt: {prompt}")
        print("-" * 80)
        for model_key, text in model_results.items():
            print(f"\n[{model_key.upper()}]")
            print(text if text else "(Generation failed)")
        print()

    return results


def main():
    """Main entry point - modify prompts and models here."""
    # Configure your settings here
    prompts = [
        "What is artificial intelligence?",
    ]

    model_keys = list(MODELS.keys())  # Use all models
    # Or specify specific models:
    # model_keys = ["llama", "mixtral"]

    # Generate text
    generate_with_models(
        prompts=prompts,
        model_keys=model_keys,
        max_length=256,
        temperature=0.7,
        top_p=0.9,
        load_in_8bit=False,
        output_file="results.json",
    )


if __name__ == "__main__":
    main()
