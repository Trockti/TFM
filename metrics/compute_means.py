import os
import json
import glob
import pandas as pd

def aplanar_diccionario(d, parent_key='', sep='___'):
    """
    Aplana el diccionario anidado para poder usar Pandas.
    Maneja correctamente valores directos, listas numéricas (SARI) 
    y listas de diccionarios (referencias de legibilidad).
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(aplanar_diccionario(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if len(v) > 0:
                # Caso 1: Es una lista de números (ej. métricas SARI: [0.4, 0.6])
                if all(isinstance(elem, (int, float)) for elem in v):
                    items.append((new_key, sum(v) / len(v)))
                
                # Caso 2: Es una lista de diccionarios (ej. referencias: [{"fernandez_huerta": 60}, {...}])
                elif all(isinstance(elem, dict) for elem in v):
                    merged_dict = {}
                    all_keys = set().union(*(elem.keys() for elem in v))
                    for sub_k in all_keys:
                        sub_vals = [elem[sub_k] for elem in v if sub_k in elem and isinstance(elem[sub_k], (int, float))]
                        if sub_vals:
                            merged_dict[sub_k] = sum(sub_vals) / len(sub_vals)
                    # Aplana el diccionario resultante del promedio de las referencias
                    items.extend(aplanar_diccionario(merged_dict, new_key, sep=sep).items())
        elif isinstance(v, (int, float)):
            items.append((new_key, v))
            
    return dict(items)

def desaplanar_diccionario(d_plano, sep='___'):
    """Reconstruye la estructura original del diccionario a partir de las claves aplanadas."""
    resultado = {}
    for k, v in d_plano.items():
        if k == 'model_name':
            continue
            
        partes = k.split(sep)
        actual = resultado
        
        # Navegar y crear sub-diccionarios
        for parte in partes[:-1]:
            if parte not in actual:
                actual[parte] = {}
            actual = actual[parte]
            
        # Asignar el valor final (convirtiéndolo a float estándar y redondeando a 4 decimales)
        try:
            val = round(float(v), 4)
        except (ValueError, TypeError):
            val = v
            
        actual[partes[-1]] = val
        
    # 1. Las métricas de SARI deben volver a convertirse en una lista: [valor_medio]
    try:
        sari = resultado.get('simplification', {}).get('sari', {})
        for metrica in ['sari_total', 'sari_add', 'sari_keep', 'sari_del']:
            if metrica in sari and not isinstance(sari[metrica], list):
                sari[metrica] = [sari[metrica]]
    except Exception:
        pass
        
    # 2. Las referencias de readability van dentro de una lista de diccionarios: [{...}]
    try:
        readability = resultado.get('readability', {})
        for estilo in ["Fernandez-Huerta", "Inflesz"]:
            sub_read = readability.get(estilo, {})
            if 'references' in sub_read and isinstance(sub_read['references'], dict):
                sub_read['references'] = [sub_read['references']]
    except Exception:
        pass
        
    return resultado

def procesar_carpetas(directorio_base="."):
    carpetas = glob.glob(os.path.join(directorio_base, "results_*"))
    
    for carpeta in carpetas:
        if not os.path.isdir(carpeta):
            continue
            
        archivos_json = glob.glob(os.path.join(carpeta, "metrics_*.json"))
        if not archivos_json:
            continue
            
        datos_carpeta = []
        print(f"Buscando archivos en: {carpeta}")
        
        for archivo in archivos_json:
            with open(archivo, 'r', encoding='utf-8') as f:
                try:
                    datos = json.load(f)
                except json.JSONDecodeError:
                    print(f"❌ Error al leer el archivo (JSON inválido): {archivo}")
                    continue
                    
            for registro in datos:
                nombre_modelo = registro.get("model_name", "desconocido")
                scores = registro.get("scores", {})
                
                # Aplanamos calculando los promedios internos de las listas de este registro específico
                scores_aplanados = aplanar_diccionario(scores)
                scores_aplanados['model_name'] = nombre_modelo
                
                datos_carpeta.append(scores_aplanados)
        
        if datos_carpeta:
            # Convertir a DataFrame de Pandas
            df = pd.DataFrame(datos_carpeta)
            
            # Calcular la media global agrupando por modelo
            df_medias = df.groupby('model_name').mean().reset_index()
            
            # Construir la estructura JSON final reconstruyendo los diccionarios anidados
            resultados_json = []
            for _, row in df_medias.iterrows():
                row_dict = row.dropna().to_dict()
                modelo = row_dict.pop('model_name')
                
                # Volver a anidar las métricas con las medias calculadas
                scores_anidados = desaplanar_diccionario(row_dict)
                
                # Mantener la estructura limpia requerida
                resultados_json.append({
                    "model_name": modelo,
                    "scores": scores_anidados
                })
            
            # Guardar el archivo JSON final de medias
            ruta_salida_json = os.path.join(carpeta, "medias_metricas_por_modelo.json")
            with open(ruta_salida_json, 'w', encoding='utf-8') as f:
                json.dump(resultados_json, f, indent=2, ensure_ascii=False)
            
            # Guardar el CSV con separador europeo listo para Excel
            ruta_salida_csv = os.path.join(carpeta, "medias_metricas_por_modelo.csv")
            df_medias.to_csv(ruta_salida_csv, index=False, sep=";", decimal=",", encoding='utf-8')
            
            print(f"✅ Procesado con éxito '{os.path.basename(carpeta)}'. Archivos guardados.\n")

if __name__ == "__main__":
    print("Iniciando el cálculo de medias y reestructuración JSON...\n")
    procesar_carpetas()
    print("\n¡Proceso finalizado con éxito!")