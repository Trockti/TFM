#!/usr/bin/env python3
"""
Program to run prompts with multiple Hugging Face language models.

Models:
1. meta-llama/Llama-3.2-3B - Multilingual Llama model
2. mistralai/Ministral-3-8B-Instruct-2512 - Ministral model
3. HiTZ/Latxa-Qwen3-VL-8B-Instruct - Spanish/Catalan/Basque specialized model
4. IIC/RigoChat-7b-v2 - Spanish language model
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "name": "meta-llama/Llama-3.2-3B",
        "description": "Multilingual Llama 3.2 3B model",
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
        prompt: str,
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
                # For chat-style VLM pipelines, send prompt as a text content block.
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                outputs = model_obj["pipeline"](
                    text=messages,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )
                return self._extract_vlm_text(outputs)

            model = model_obj["model"]
            tokenizer = model_obj["tokenizer"]

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            return tokenizer.batch_decode(output_ids, skip_special_tokens=True)
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
    prompts: List[str],
    model_key: str,
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    load_in_8bit: bool = False,
) -> Dict[str, Optional[str]]:
    """Generate text using a specified model and prompts."""
    manager = ModelManager(device=device, cache_dir=cache_dir)
    
    results: Dict[str, Optional[str]] = {p: None for p in prompts}

    logger.info("\n--- Loading model: %s ---", model_key)
    if not manager.load_model(model_key, load_in_8bit=load_in_8bit):
        logger.warning("Could not load model: %s", model_key)
        return results

    for i, prompt in enumerate(prompts, 1):
        logger.info("Processing prompt %d/%d with %s...", i, len(prompts), model_key)
        texts = manager.generate(
            model_key,
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )
        if texts:
            results[prompt] = texts[0]
            logger.info("    Generated %d characters", len(texts[0]))
        else:
            logger.warning("    Generation failed")
            
    # Unload model to free memory for the next one
    manager.unload_model(model_key)

    return results


def main() -> None:
    """Main entry point - modify prompts and models here."""
    prompts = [
        "What is artificial intelligence?",
    ]

    model_keys = list(MODELS.keys())
    
    # Initialize overall results structure
    all_results: Dict[str, Dict[str, Optional[str]]] = {p: {} for p in prompts}

    for model_key in model_keys:
        model_results = generate_with_model(
            prompts=prompts,
            model_key=model_key,
            max_length=256,
            temperature=0.7,
            top_p=0.9,
            load_in_8bit=False,
        )
        for prompt, text in model_results.items():
            all_results[prompt][model_key] = text

    output_file = "results.json"
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(all_results, file_handle, indent=2, ensure_ascii=False)
        logger.info("Results saved to: %s", output_path)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for prompt, model_results in all_results.items():
        print(f"\nPrompt: {prompt}")
        print("-" * 80)
        for model_key, text in model_results.items():
            print(f"\n[{model_key.upper()}]")
            print(text if text else "(Generation failed)")
        print()

if __name__ == "__main__":
    main()
