import os
import json
import glob
import pandas as pd

def aplanar_diccionario(d, parent_key='', sep='___'):
    """Aplana el diccionario anidado para poder usar Pandas."""
    items = []
    for k, v in d.items():
        # Usamos '___' como separador seguro para no chocar con guiones bajos normales
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(aplanar_diccionario(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if len(v) > 0:
                if isinstance(v, dict):
                    items.extend(aplanar_diccionario(v, new_key, sep=sep).items())
                elif isinstance(v, (int, float)):
                    items.append((new_key, sum(v) / len(v)))
        elif isinstance(v, (int, float)):
            items.append((new_key, v))
            
    return dict(items)

def desaplanar_diccionario(d_plano, sep='___'):
    """Reconstruye el diccionario anidado a partir de las claves aplanadas."""
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
        
    try:
        sari = resultado.get('simplification', {}).get('sari', {})
        for metrica in ['sari_total', 'sari_add', 'sari_keep', 'sari_del']:
            if metrica in sari:
                sari[metrica] = [sari[metrica]]
    except Exception:
        pass
        
    # 2. Las referencias de readability van dentro de una lista de diccionarios: [{...}]
    try:
        readability = resultado.get('readability', {})
        if 'references' in readability and isinstance(readability['references'], dict):
            readability['references'] = [readability['references']]
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
        
        for archivo in archivos_json:
            with open(archivo, 'r', encoding='utf-8') as f:
                try:
                    datos = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error al leer el archivo (JSON inválido): {archivo}")
                    continue
                    
            for registro in datos:
                nombre_modelo = registro.get("model_name", "desconocido")
                scores = registro.get("scores", {})
                
                scores_aplanados = aplanar_diccionario(scores)
                scores_aplanados['model_name'] = nombre_modelo
                
                datos_carpeta.append(scores_aplanados)
        
        if datos_carpeta:
            # Calcular la media
            df = pd.DataFrame(datos_carpeta)
            df_medias = df.groupby('model_name').mean().reset_index()
            
            # Construir la estructura JSON final
            resultados_json = []
            for _, row in df_medias.iterrows():
                row_dict = row.dropna().to_dict()
                modelo = row_dict.pop('model_name')
                
                # Volver a anidar las métricas
                scores_anidados = desaplanar_diccionario(row_dict)
                
                # Mantener la estructura principal
                resultados_json.append({
                    "model_name": modelo,
                    "scores": scores_anidados
                })
            
            # Guardar el JSON 
            ruta_salida_json = os.path.join(carpeta, "medias_metricas_por_modelo.json")
            with open(ruta_salida_json, 'w', encoding='utf-8') as f:
                json.dump(resultados_json, f, indent=2, ensure_ascii=False)
            
            # Opcional: seguimos generando el CSV por si quieres verlo rápido en Excel
            ruta_salida_csv = os.path.join(carpeta, "medias_metricas_por_modelo.csv")
            df_medias.to_csv(ruta_salida_csv, index=False, encoding='utf-8')
            
            print(f"✅ Procesado '{os.path.basename(carpeta)}'. Archivos guardados.")

if __name__ == "__main__":
    print("Iniciando el cálculo de medias y reestructuración JSON...\n")
    procesar_carpetas()
    print("\n¡Proceso finalizado!")