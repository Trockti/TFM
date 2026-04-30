from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
import ollama

import os
os.environ["DEEPEVAL_VERBOSE"] = "True"

# 1. Define the Custom Gemma 4 Judge
class Gemma4Judge(DeepEvalBaseLLM):
    def __init__(self, model_name="gemma4:latest"):
        self.model_name = model_name

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# 2. Initialize the Judge and Metric
gemma_judge = Gemma4Judge()
metric = FaithfulnessMetric(
    threshold=0.7, 
    model=gemma_judge, 
    include_reason=True
)

# 3. Your Spanish Data
original_text = """
El Real Madrid ganó su 15ª Champions League en 2024 tras derrotar al 
Borussia Dortmund por 2-0 en Wembley. Los goles fueron de Carvajal y Vinícius.
"""

generated_text = """
El Madrid consiguió la 15ª en 2024 contra el Dortmund. 
El partido fue en París y los goles los metió Rodrygo.
"""

# 4. Create and Measure the Test Case
test_case = LLMTestCase(
    input="¿Qué pasó en la final de la Champions 2024?",
    actual_output=generated_text,
    retrieval_context=[original_text]
)

metric.measure(test_case)

# 5. Result: Identifying the specific hallucinations
print(f"Puntuación (Fidelidad): {metric.score}")
print("\n--- ANÁLISIS DE SEGMENTOS ---")

# Comprobamos que el métrico haya generado las afirmaciones (claims)
if hasattr(metric, 'claims') and len(metric.claims) == len(metric.verdicts):
    
    # Unimos cada afirmación (claim) con su veredicto correspondiente
    for claim, verdict in zip(metric.claims, metric.verdicts):
        is_hallucination = verdict.verdict.lower() == "no"
        status = "❌ ALUCINACIÓN" if is_hallucination else "✅ OK"
        
        # 1. Imprimimos el estado y la frase exacta extraída del texto
        print(f"\n{status}: {claim}")
        
        # 2. Imprimimos el motivo del modelo (siempre que lo haya dado)
        if verdict.reason:
            print(f"   Motivo de Gemma 4: {verdict.reason}")

else:
    # Fallback de seguridad en caso de que las listas no coincidan
    for verdict in metric.verdicts:
        is_hallucination = verdict.verdict.lower() == "no"
        status = "❌ ALUCINACIÓN" if is_hallucination else "✅ OK"
        
        if is_hallucination:
            print(f"\n{status}: {verdict.reason}")
        else:
            print(f"\n{status}: (Segmento verificado correctamente)")