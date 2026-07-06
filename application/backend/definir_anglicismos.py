import os
import json
import difflib
import re
from dotenv import load_dotenv
from huggingface_hub import login
import textstat

# Importar funciones específicas de tus scripts locales
from hulat_metrics import calculate_bertscore, LANG_CFG
from use_models import generate_with_model
from analize_definitions import analyze_text  

# Cargar variables de entorno
load_dotenv()

# Obtener y configurar token de Hugging Face
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None:
    raise ValueError("No se encontró el token de Hugging Face. Asegúrate de configurar la variable de entorno HUGGINGFACE_TOKEN.")
login(token=hf_token)


def normalizar_termino(termino):
    """
    Convierte a minúsculas y elimina espacios y guiones para comparar.
    """
    return re.sub(r'[\s\-]', '', termino.lower())


def buscar_en_dataset(palabra, ruta_json="./data/normalized_definitions.json"):
    """
    Busca la palabra en el archivo JSON.
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

    for termino_dataset, definicion in datos.items():
        # 1. Coincidencia con el término completo en bruto (por si acaso)
        if normalizar_termino(termino_dataset) == palabra_norm:
            return definicion
            
        # 2. Si el término tiene un formato como "pow (proof of work)"
        if " (" in termino_dataset and termino_dataset.endswith(")"):
            # Dividimos en un máximo de 2 partes
            partes = termino_dataset.split(" (", 1) 
            
            termino_acronimo = partes[0].strip()                # Extrae "pow"
            termino_completo = partes[1].replace(")", "").strip() # Extrae "proof of work"
            
            # Comprobamos si la palabra coincide con ALGUNA de las dos partes
            if normalizar_termino(termino_acronimo) == palabra_norm or normalizar_termino(termino_completo) == palabra_norm:
                return definicion
        else:
            # Si no tiene paréntesis, comprobamos el término normal
            if normalizar_termino(termino_dataset) == palabra_norm:
                return definicion
            
    try:
        with open("./data/definitions_v2.json", 'r', encoding='utf-8') as f:
            datos_v2 = json.load(f)
            terminos_disponibles = list(datos_v2.keys())
            coincidencias = difflib.get_close_matches(palabra.lower(), terminos_disponibles, n=1, cutoff=0.9)
            
            if coincidencias:
                # Extraemos el primer elemento de la lista con [0]
                return datos_v2[coincidencias[0]] 
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return None


def check_metrics(texto_generado, palabra, referencia_dataset=None):
    """
    Evalúa el texto generado contra los umbrales establecidos:
    - Fernandez-Huerta >= 70
    - Compliance Score >= 0.7
    - BERTScore >= 0.6 (solo si existe una referencia en el diccionario)
    """
    fh_score = round(textstat.fernandez_huerta(texto_generado), 4)
    
    analysis = analyze_text(texto_generado, term=palabra)
    comp_score = analysis.get("Prompt Compliance Score", 0.0)
    
    bert_f1 = None
    passes = 0
    thresholds_total = 2  
    
    if fh_score >= 70.0:
        passes += 1
        
    if comp_score >= 0.7:
        passes += 1
        
    if referencia_dataset:
        thresholds_total = 3
        cfg = LANG_CFG["es"]
        _, _, bert_f1 = calculate_bertscore(texto_generado, referencia_dataset, cfg)
        bert_f1 = round(bert_f1, 4)
        if bert_f1 >= 0.6:
            passes += 1

    return {
        "text": texto_generado,
        "fh_score": fh_score,
        "comp_score": comp_score,
        "bert_f1": bert_f1,
        "passes": passes,
        "all_passed": passes == thresholds_total
    }


def definir_anglicismo(palabra, frase=None, ruta_dataset="./data/normalized_definitions.json"):
    """
    Genera la definición de un anglicismo utilizando modelos y los nuevos prompts estructurados.
    """
    definicion_dataset = buscar_en_dataset(palabra, ruta_dataset)
    
    if definicion_dataset:
        print(f"Definición encontrada en dataset: {definicion_dataset}\n")
        print(f"¡Se encontró '{palabra}' en el dataset local!")
    else:
        print("No se encontró la palabra en el dataset local.")
        
    # Manejar el contexto por si viene vacío
    contexto_str = frase if frase else "Sin contexto específico."
    
    # Bloque dinámico para el Prompt 2
    referencia_bloque = f"Referencia terminológica:\n{definicion_dataset}\n\n" if definicion_dataset else ""

    # ==========================================
    # INTENTO 1: Zero-Shot (Basado en reglas)
    # ==========================================
    prompt_1 = [
        {
            "role": "system",
            "content": "Eres un especialista en lectura fácil, accesibilidad cognitiva y lenguaje claro. Creas glosas breves para explicar anglicismos en español."
        },
        {
            "role": "user",
            # Se usa f-string para inyectar {palabra} y {contexto_str}
            "content": f"Crea una glosa breve para el término: {palabra}\n\nContexto:\n{contexto_str}\n\nInstrucciones:\n- Explica el significado del término en este contexto.\n- Usa palabras comunes, concretas y frecuentes.\n- Usa frases cortas.\n- Usa voz activa y orden directo.\n- No hagas una explicación larga.\n- No inventes información.\n- No uses el término para definirse a sí mismo.\n- Usa 1 frase si es posible.\n- Usa 2 frases solo si mejora la comprensión.\n- Máximo recomendado: 25 palabras.\n- Empieza indicando qué tipo de cosa es el término y después explica su función o significado principal.\n- Puedes usar estructuras como: \"aparato que...\", \"persona que...\", \"actividad que...\", \"tecnología que...\", \"sistema que...\", \"forma de...\".\n\nSalida:\nGlosa: ..."
        }
    ]

    
    print("--- GENERANDO INTENTO 1 (Zero-Shot) ---")
    res_1 = generate_with_model(prompts=[prompt_1], model_key="latxa", max_length=256, temperature=0.2, top_p=0.9, device="cuda")
    
    texto_1 = res_1[0] if (res_1 and res_1[0] is not None) else ""
    print(f"Texto generado en intento 1:\n{texto_1}\n")
    eval_1 = check_metrics(texto_1, palabra, definicion_dataset)
    print(f"Evaluación intento 1:\nFernandez-Huerta: {eval_1['fh_score']}\nCompliance Score: {eval_1['comp_score']}\nBERTScore F1: {eval_1['bert_f1']}\nPasó umbrales: {eval_1['all_passed']}\n")
    if eval_1["all_passed"]:
        return eval_1["text"]

    # ==========================================
    # INTENTO 2: Few-Shot (Basado en ejemplos y referencia)
    # ==========================================
    prompt_2 = [
        {
            "role": "system",
            "content": "Eres un especialista en lectura fácil, accesibilidad cognitiva y lenguaje claro. Creas glosas breves siguiendo el estilo de un diccionario fácil."
        },
        {
            "role": "user",
            # Se usa f-string para inyectar variables y el bloque de referencia dinámico
            "content": f"Ejemplo 1:\nTérmino: Router\nContexto: El router de casa no funciona bien.\nReferencia terminológica: Dispositivo que distribuye conexiones de red entre varios equipos.\nGlosa: Aparato que permite conectar varios dispositivos a internet. Es una palabra inglesa y se pronuncia \"ruter\".\n\nEjemplo 2:\nTérmino: Hobby\nContexto: La fotografía es su hobby favorito.\nReferencia terminológica: Actividad que una persona realiza por gusto en su tiempo libre.\nGlosa: Actividad que una persona hace en su tiempo libre porque le gusta.\n\nEjemplo 3:\nTérmino: WiFi\nContexto: El WiFi del hotel no funciona.\nReferencia terminológica: Tecnología inalámbrica para conectar dispositivos a internet.\nGlosa: Tecnología que permite conectar dispositivos a internet sin usar cables.\n\nAhora crea una glosa para:\n\nTérmino: {palabra}\nContexto:\n{contexto_str}\n\n{referencia_bloque}Instrucciones:\n- Sigue el mismo estilo de los ejemplos.\n- Usa palabras comunes y concretas.\n- Usa frases breves.\n- Usa 1 frase si es posible.\n- Usa 2 frases solo si mejora la comprensión.\n- Máximo recomendado: 25 palabras.\n- No hagas una explicación enciclopédica.\n- No inventes información.\n- No uses el término para definirse a sí mismo.\n\nSalida:\nGlosa: ..."
        }
    ]


    print("--- GENERANDO INTENTO 2 (Few-Shot) ---")
    res_2 = generate_with_model(prompts=[prompt_2], model_key="latxa", max_length=256, temperature=0.7, top_p=0.9, device="cuda")
    texto_2 = res_2[0] if (res_2 and res_2[0] is not None) else ""
    eval_2 = check_metrics(texto_2, palabra, definicion_dataset)

    if eval_2["all_passed"]:
        return eval_2["text"]

    # ==========================================
    # LÓGICA DE RESOLUCIÓN DE EMPATES
    # ==========================================
    if eval_1["passes"] == 0 and eval_2["passes"] == 0:
        if definicion_dataset:
            return definicion_dataset
        else:
            return "El sistema no fue capaz de generar una definición para la palabra, pruebe de nuevo."

    if eval_1["passes"] > eval_2["passes"]:
        return eval_1["text"]
    elif eval_2["passes"] > eval_1["passes"]:
        return eval_2["text"]
        
    if eval_1["comp_score"] >= eval_2["comp_score"]:
        return eval_1["text"]
    else:
        return eval_2["text"]


# Bloque de prueba
if __name__ == "__main__":
    ejemplo = "AI"
    frase = "La AI está revolucionando la industria tecnológica."
    
    print(f"Buscando definición para: {ejemplo}\n" + "-"*40)
    definicion = definir_anglicismo(ejemplo, frase, "./data/normalized_definitions.json")
    print("\nRESULTADO FINAL:\n", definicion)