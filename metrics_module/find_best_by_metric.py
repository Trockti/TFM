import os
import json
import glob
import pandas as pd

def flatten_dict(d, parent_key='', sep='___'):
    """Aplana el diccionario anidado para facilitar el análisis con Pandas."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if len(v) > 0:
                if isinstance(v, dict):
                    # Promediar lista de diccionarios (ej. referencias de legibilidad)
                    merged_dict = {}
                    all_keys = set().union(*(elem.keys() for elem in v))
                    for sub_k in all_keys:
                        sub_vals = [elem[sub_k] for elem in v if sub_k in elem and isinstance(elem[sub_k], (int, float))]
                        if sub_vals:
                            merged_dict[sub_k] = sum(sub_vals) / len(sub_vals)
                    items.extend(flatten_dict(merged_dict, new_key, sep=sep).items())
                elif isinstance(v, (int, float)):
                    # Promediar lista de números (ej. métricas SARI)
                    items.append((new_key, sum(v) / len(v)))
        elif isinstance(v, (int, float)):
            items.append((new_key, v))
            
    return dict(items)

def process_specific_metrics_per_folder(base_dir="."):
    # Definir exactamente las métricas que queremos buscar y cómo queremos que se llamen en el reporte
    metricas_objetivo = {
        'similarity___bertscore___bertscore_f1': 'BERTScore F1',
        'similarity___meaningbert': 'MeaningBERT',  # <--- AÑADIDO
        'similarity___bleu___bleu': 'BLEU',
        'similarity___rouge___rouge-l_f1': 'ROUGE-L F1',
        'similarity___rouge___rouge-1_f1': 'ROUGE-1 F1',
        'similarity___rouge___rouge-2_f1': 'ROUGE-2 F1',
        'factuality___alignscore': 'AlignScore',
        'factuality___questeval': 'QuestEval',
        'factuality___summac___summac_zs': 'SummaC ZS',
        'factuality___summac___summac_conv': 'SummaC Conv',
        # Rutas de legibilidad corregidas para coincidir con la estructura JSON real:
        'readability___Fernandez-Huerta___simplified___fernandez_huerta': 'Fernandez Huerta (Simplified)',
        'readability___Inflesz___simplified___inflesz': 'Inflesz (Simplified)'  # <--- AÑADIDO
    }

    folders = glob.glob(os.path.join(base_dir, "experiments", "results_*"))
    
    for folder in folders:
        if not os.path.isdir(folder):
            continue
            
        json_files = glob.glob(os.path.join(folder, "metrics_*.json"))
        if not json_files:
            continue
            
        print(f"\nProcesando la carpeta: '{os.path.basename(folder)}'...")
        folder_data = []
        
        # 1. Leer todas las métricas
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"  [!] Saltando JSON inválido: {file}")
                    continue
                    
            for record in data:
                model_name = record.get("model_name", "unknown")
                scores = record.get("scores", {})
                
                flat_scores = flatten_dict(scores)
                flat_scores['model_name'] = model_name
                
                folder_data.append(flat_scores)
        
        if not folder_data:
            print(f"  [-] No se encontraron datos válidos en {folder}")
            continue

        # 2. Calcular la media agrupada por modelo
        df = pd.DataFrame(folder_data)
        df_means = df.groupby('model_name').mean()
        
        # 3. Filtrar y encontrar el mejor modelo SOLO para las métricas objetivo
        best_results = []
        
        for clave_interna, nombre_legible in metricas_objetivo.items():
            if clave_interna in df_means.columns:
                best_model = df_means[clave_interna].idxmax()
                best_score = df_means[clave_interna].max()
                
                best_results.append({
                    "Metric": nombre_legible,
                    "Best_Model": best_model,
                    "Mean_Score": round(best_score, 4)
                })
            else:
                print(f"  [!] Advertencia: La métrica '{clave_interna}' no se encontró en los datos de esta carpeta.")
            
        # 4. Generar y guardar los reportes
        if best_results:
            df_best = pd.DataFrame(best_results)
            
            # Guardar CSV (Añadido punto y coma y coma decimal para formato europeo)
            csv_path = os.path.join(folder, "best_models_by_metric.csv")
            df_best.to_csv(csv_path, index=False, sep=";", decimal=",", encoding='utf-8')
            
            # Guardar JSON
            json_path = os.path.join(folder, "best_models_by_metric.json")
            df_best.to_json(json_path, orient='records', indent=4, force_ascii=False)
            
            print(f"  ✅ Archivos guardados:")
            print(f"     - {csv_path}")
            print(f"     - {json_path}")

if __name__ == "__main__":
    print("Buscando los mejores modelos por métricas específicas...")
    process_specific_metrics_per_folder()
    print("\n¡Proceso finalizado!")