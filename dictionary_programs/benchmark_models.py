import os
import json
import extract_definitions
import sys
import shutil

# Define models
MODELS = [
    "llama3.1:8b",
    "deepseek-r1:8b",
    "mistral:7b",
    "gemma3:12b",
]

def evaluate_extraction(extracted_file, ground_truth_file):
    """
    Evaluates the extraction against the ground truth.
    Returns a dictionary of metrics.
    """
    # Load Expected Definitions
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
        
    # Load Extracted Data
    if not os.path.exists(extracted_file):
        print(f"Extracted file not found: {extracted_file}")
        return None
        
    with open(extracted_file, 'r', encoding='utf-8') as f:
        extracted = json.load(f)
        
    # Prepare comparison dictionary
    extracted_lower = {k.lower().strip(): v for k, v in extracted.items()}
    
    missing_words = []
    wrong_definitions = []
    total_words = len(ground_truth)
    
    for word, expected_def in ground_truth.items():
        # Clean up the key for lookup
        word_key = word.lower().strip()
        
        if word_key not in extracted_lower:
            missing_words.append(word)
        else:
            actual_def = extracted_lower[word_key]
            # Compare definitions (normalizing stripping whitespace)
            if actual_def.strip() != expected_def.strip():
                wrong_definitions.append({
                    'word': word,
                    'expected': expected_def,
                    'actual': actual_def
                })
                
    # Calculate requested metrics
    missing_count = len(missing_words)
    wrong_def_count = len(wrong_definitions)
    total_failures = missing_count + wrong_def_count
    
    failure_rate = (total_failures / total_words * 100) if total_words > 0 else 0
    wrong_def_rate = (wrong_def_count / total_words * 100) if total_words > 0 else 0
    
    return {
        "total_words": total_words,
        "missing_count": missing_count,
        "wrong_def_count": wrong_def_count,
        "failure_rate": failure_rate,
        "wrong_def_rate": wrong_def_rate
    }

def load_failure_rates(failure_rates_file):
    """
    Load historical failure rates from file.
    Returns dictionary with model names as keys and lists of failure rates as values.
    """
    if os.path.exists(failure_rates_file):
        with open(failure_rates_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_failure_rates(failure_rates_file, failure_rates):
    """
    Save failure rates tracking to file.
    """
    with open(failure_rates_file, 'w', encoding='utf-8') as f:
        json.dump(failure_rates, f, indent=2)

def main():
    results = {}
    failure_rates_history = {}
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "../data/diccionario_anglicismos.pdf")
    ground_truth_path = os.path.join(base_dir, "../tests/test_definitions_cervantes.json")
    output_dir = os.path.join(base_dir, "../data/benchmark_results")
    failure_rates_file = os.path.join(output_dir, "failure_rates_history.json")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load existing failure rates history
    failure_rates_history = load_failure_rates(failure_rates_file)

    print(f"Starting benchmark for {len(MODELS)} models...")
    print(f"PDF Path: {pdf_path}")
    print(f"Ground Truth: {ground_truth_path}")

    for model in MODELS:
        print(f"\n==========================================")
        print(f"Testing model: {model}")
        print(f"==========================================")
        
        sanitized_model = model.replace(":", "_").replace(".", "_")
        extracted_file = os.path.join(output_dir, f"extracted_{sanitized_model}.json")
        model_results_file = os.path.join(output_dir, f"results_{sanitized_model}.json")
        
        try:
            # 1. Run Extraction
            print(f"Extracting definitions...")
            extracted_data = extract_definitions.process_pdf(pdf_path, model_name=model)
            
            # Save extracted data
            with open(extracted_file, "w", encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            print(f"Extraction saved to: {extracted_file}")
            
            # 2. Evaluate
            print(f"Evaluating...")
            metrics = evaluate_extraction(extracted_file, ground_truth_path)
            
            if metrics:
                results[model] = metrics
                print(f"Results for {model}:")
                print(f"  Failure Rate: {metrics['failure_rate']:.2f}%")
                print(f"  Wrong Definition Rate: {metrics['wrong_def_rate']:.2f}%")
                
                # Save individual model results file
                with open(model_results_file, "w", encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2)
                print(f"Results saved to: {model_results_file}")
                
                # Update failure rates history
                if model not in failure_rates_history:
                    failure_rates_history[model] = []
                failure_rates_history[model].append(metrics['failure_rate'])
                
                # Save updated failure rates history
                save_failure_rates(failure_rates_file, failure_rates_history)
                print(f"Failure rate history updated for {model}")
                
            else:
                results[model] = {"error": "Evaluation failed"}
                
        except Exception as e:
            print(f"Error processing model {model}: {e}")
            results[model] = {"error": str(e)}

    # Save final benchmark results
    benchmark_file = os.path.join(output_dir, "benchmark_summary.json")
    with open(benchmark_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n==========================================")
    print(f"Benchmark Complete. Results saved to {benchmark_file}")
    
    # Display failure rates history
    print(f"\n==========================================")
    print(f"Failure Rates History (all runs):")
    print(f"==========================================")
    for model, rates in failure_rates_history.items():
        avg_rate = sum(rates) / len(rates) if rates else 0
        print(f"{model}:")
        print(f"  All rates: {[f'{r:.2f}%' for r in rates]}")
        print(f"  Average: {avg_rate:.2f}%")
        print(f"  Latest: {rates[-1]:.2f}%" if rates else f"  Latest: N/A")
    
    # Find best model
    best_model = None
    min_failure_rate = float('inf')
    
    for model, metrics in results.items():
        if "failure_rate" in metrics:
            if metrics["failure_rate"] < min_failure_rate:
                min_failure_rate = metrics["failure_rate"]
                best_model = model
    
    if best_model:
        print(f"\n==========================================")
        print(f"Best Model (this run): {best_model} with Failure Rate: {min_failure_rate:.2f}%")
        print(f"==========================================")
        
        # Save best results
        sanitized_best_model = best_model.replace(":", "_").replace(".", "_")
        best_model_file = os.path.join(output_dir, f"extracted_{sanitized_best_model}.json")
        best_results_file = os.path.join(base_dir, "../data/extracted_definitions_best.json")
        
        if os.path.exists(best_model_file):
            shutil.copy2(best_model_file, best_results_file)
            print(f"Saved best results to: {best_results_file}")
        else:
            print(f"Error: Could not find extracted file for best model at {best_model_file}")
    else:
        print("\nCould not determine best model.")

if __name__ == "__main__":
    main()
