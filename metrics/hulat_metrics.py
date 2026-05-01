# ****** Librerías estándar ******
import os
import sys
import re
import csv
import json
import nltk
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from docx import Document
from pathlib import Path
from huggingface_hub import login
from collections import defaultdict

# ****** Librerías externas ******
import torch
import pyphen
import evaluate
import textstat
import bleu
import summac.summac.model_summac as summac_mod
from bert_score import score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from rouge import Rouge
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# **** Métricas especializadas ****
from AlignScore_v2_es.src.alignscore import AlignScore
from moverscore.moverscore_v2 import get_idf_dict, word_mover_score

from QuestEval.questeval.questeval_metric import QuestEval
from summac.summac.model_summac import SummaCZS, SummaCConv
from codecarbon import OfflineEmissionsTracker

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('stopwords', quiet=True)

BASE_DIR = Path(__file__).resolve().parent

# ***************************
# ****** CONFIGURACIÓN ******
# ***************************
LANG_CFG = {
    "es": {
        "bertscore_model": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "bertscore_lang": "es",
        "moverscore_model": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "alignscore_model": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "alignscore_eval_mode": "nli_sp",
        "alignscore_ckpt_path": None, # No se usa en este caso
        "summac_key": "mdeberta-es",
        "summac_model_card": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "readability": "fh",
    },
    "en": {
        "bertscore_model": "roberta-large",
        "bertscore_lang": "en",
        "moverscore_model": "roberta-large",
        "alignscore_model": "roberta-base",
        "alignscore_eval_mode": "nli_sp",
        "alignscore_ckpt_path": str(BASE_DIR / "AlignScore_v2_es" / "checkpoint" / "checkpoints" / "AlignScore-base.ckpt"),
        "summac_key": "tals/albert-xlarge-vitaminc-mnli",
        "summac_model_card": "tals/albert-xlarge-vitaminc-mnli",
        "readability": "flesch",
    }
}


# ****** INICIALIZACIÓN DE SUMMAC ******
def init_summac(cfg):
    spanish_nli_config = {
        "model_card": cfg["summac_model_card"],
        "entailment_idx": 0,
        "contradiction_idx": 2,
    }
    summac_mod.model_map[cfg["summac_key"]] = spanish_nli_config

    print("Modelo registrado en SummaC:")
    print(summac_mod.model_map[cfg["summac_key"]])

    model_zs = SummaCZS(granularity="sentence", model_name=cfg["summac_key"], device="cuda")
    model_conv = SummaCConv(models=[cfg["summac_key"]], bins="percentile", granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
    
    return model_zs, model_conv

# ****** INICIALIZACIÓN DE ALIGNSCORE ******
def init_alignscore(cfg, device="cuda:0", batch_size=32):
    ckpt = cfg.get("alignscore_ckpt_path", None)
    scorer = AlignScore(model=cfg["alignscore_model"], batch_size=batch_size, ckpt_path=ckpt, device=device, evaluation_mode=cfg["alignscore_eval_mode"])
    return scorer

# ****** FUNCIONES DE MÉTRICAS ******

# -----------------------
# ------ BERTScore ------
# -----------------------
def calculate_bertscore(simplified, target_text, cfg):
    if not simplified.strip() or not target_text.strip():
        return 0.0, 0.0, 0.0
    P, R, F1 = score([simplified], [target_text], model_type=cfg["bertscore_model"], num_layers=12, lang=cfg["bertscore_lang"], rescale_with_baseline=False)
    return P.mean().item(), R.mean().item(), F1.mean().item()

# ------------------------------
# ------ Readability ------
# ------------------------------
def calculate_readability(text, cfg):
    if not text.strip():
        return {}
        
    if cfg["readability"] == "fh":
        return {"fernandez_huerta": round(textstat.fernandez_huerta(text), 4)}
    elif cfg["readability"] == "flesch":
        return {
            "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 4),
        }
    else:
        return {}

# ------------------
# ------ SARI ------
# ------------------
def calculate_sari(source, prediction, references):
    def ngram_counter(sentence, n):
        return [tuple(sentence[i:i+n]) for i in range(len(sentence)-n+1)]

    def precision_recall_f1(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1

    source_tokens = source.split()
    prediction_tokens = prediction.split()
    references_tokens = [ref.split() for ref in references]

    ngram_levels = [1, 2, 3, 4]
    add_scores, keep_scores, delete_scores = [], [], []

    for n in ngram_levels:
        source_ngrams = set(ngram_counter(source_tokens, n))
        pred_ngrams = set(ngram_counter(prediction_tokens, n))
        ref_ngrams = [set(ngram_counter(ref, n)) for ref in references_tokens]

        union_ref_ngrams = set().union(*ref_ngrams)

        keep_ngrams = source_ngrams.intersection(pred_ngrams)
        add_ngrams = pred_ngrams.difference(source_ngrams)
        del_ngrams = source_ngrams.difference(pred_ngrams)

        tp_keep = len(keep_ngrams.intersection(union_ref_ngrams))
        fp_keep = len(keep_ngrams.difference(union_ref_ngrams))
        fn_keep = len(source_ngrams.intersection(union_ref_ngrams).difference(keep_ngrams))

        tp_add = len(add_ngrams.intersection(union_ref_ngrams))
        fp_add = len(add_ngrams.difference(union_ref_ngrams))
        fn_add = len(union_ref_ngrams.difference(source_ngrams).difference(add_ngrams))

        good_deletions = source_ngrams.difference(union_ref_ngrams)
        tp_del = len(del_ngrams.intersection(good_deletions))
        fp_del = len(del_ngrams.difference(good_deletions))

        keep_precision, keep_recall, keep_f1 = precision_recall_f1(tp_keep, fp_keep, fn_keep)
        add_precision, add_recall, add_f1 = precision_recall_f1(tp_add, fp_add, fn_add)
        del_precision = tp_del / (tp_del + fp_del) if (tp_del + fp_del) > 0 else 0
        del_f1 = del_precision

        add_scores.append(add_f1)
        keep_scores.append(keep_f1)
        delete_scores.append(del_f1)

    avg_add = sum(add_scores) / len(add_scores)
    avg_keep = sum(keep_scores) / len(keep_scores)
    avg_del = sum(delete_scores) / len(delete_scores)
    sari = (avg_add + avg_keep + avg_del) / 3

    return avg_add, avg_keep, avg_del, sari

# --------------------------------------------------------------------
# ------ Semantic_Answer_Similarity (SAS sentence-transformers) ------
# --------------------------------------------------------------------
def SAS(simplified, target_text, st_model):
    if not simplified.strip() or not target_text.strip():
        return 0.0
    emb_simplified = st_model.encode([simplified], convert_to_tensor=True)
    emb_target = st_model.encode([target_text], convert_to_tensor=True)
    return util.cos_sim(emb_simplified, emb_target).item() 

# ------------------------
# ------ AlignScore ------
# ------------------------
def Align_Score(original, simplified, align_scorer):
    if not simplified.strip() or not original.strip():
        return 0.0
    s = align_scorer.score(contexts=[original], claims=[simplified])
    if isinstance(s, (list, tuple, np.ndarray)):
        return float(s[0])
    return float(s)

# ------------------------------------------
# ------ Factual fidelity (QuestEval) ------
# ------------------------------------------
def Quest_Eval(original, simplified):
    if not simplified.strip() or not original.strip():
        return {"corpus_score": 0.0}
    questeval = QuestEval(no_cuda=False)
    score = questeval.corpus_questeval(
        hypothesis=[simplified],
        sources=[original]
    )
    return score

# ------------------------
# ------ MoverScore ------
# ------------------------
def calculate_moverscore(simplified, target_texts, idf_dict_ref, idf_dict_hyp, cfg, use_stopwords=False):
    if not simplified.strip() or not all(t.strip() for t in target_texts):
        return [0.0] * len(target_texts)
        
    try:
        os.environ['MOVERSCORE_MODEL'] = cfg["moverscore_model"]
        if isinstance(target_texts, str):
            target_texts = [target_texts]

        scores = word_mover_score(
            target_texts,                      
            [simplified] * len(target_texts),  
            idf_dict_ref,
            idf_dict_hyp,
            n_gram=1,
            remove_subwords=False
        )
        return [float(s) for s in scores]
    except Exception as e:
        print(f"⚠️ Error en MoverScore: {e}")
        return [0.0] * len(target_texts)

# ------------------
# ------ BLEU ------
# ------------------
def calculate_bleu(simplified, target_texts):
    bleu_metric = evaluate.load("bleu")
    
    # Manejar textos vacíos
    if not simplified.strip() or not all(t.strip() for t in target_texts):
        empty_res = [0.0] * len(target_texts)
        return {
            "bleu": empty_res, "bleu_precision-1": empty_res, "bleu_precision-2": empty_res,
            "bleu_precision-3": empty_res, "bleu_precision-4": empty_res,
            "bleu_brevity_penalty": empty_res, "bleu_length_ratio": empty_res,
            "bleu_translation_length": empty_res, "bleu_reference_length": empty_res
        }

    bleu_scores = {
        "bleu": [], "bleu_precision-1": [], "bleu_precision-2": [],
        "bleu_precision-3": [], "bleu_precision-4": [], "bleu_brevity_penalty": [],
        "bleu_length_ratio": [], "bleu_translation_length": [], "bleu_reference_length": [],
    }

    for text in target_texts:
        results = bleu_metric.compute(predictions=[simplified], references=[[text]])
        bleu_scores["bleu"].append(results["bleu"])
        bleu_scores["bleu_precision-1"].append(results["precisions"][0])
        bleu_scores["bleu_precision-2"].append(results["precisions"][1])
        bleu_scores["bleu_precision-3"].append(results["precisions"][2])
        bleu_scores["bleu_precision-4"].append(results["precisions"][3])
        bleu_scores["bleu_brevity_penalty"].append(results["brevity_penalty"])
        bleu_scores["bleu_length_ratio"].append(results["length_ratio"])
        bleu_scores["bleu_translation_length"].append(results["translation_length"])
        bleu_scores["bleu_reference_length"].append(results["reference_length"])
    return bleu_scores

# -------------------
# ------ ROUGE ------
# -------------------
def calculate_rouge(simplified, target_texts):
    rouge = Rouge()
    
    if not simplified.strip() or not all(t.strip() for t in target_texts):
        empty_res = [0.0] * len(target_texts)
        return {
            "rouge-1": {"rouge-1_f1": empty_res, "rouge-1_precision": empty_res, "rouge-1_recall": empty_res},
            "rouge-2": {"rouge-2_f1": empty_res, "rouge-2_precision": empty_res, "rouge-2_recall": empty_res},
            "rouge-l": {"rouge-l_f1": empty_res, "rouge-l_precision": empty_res, "rouge-l_recall": empty_res}
        }

    rouge_scores = {
        "rouge-1": {"rouge-1_f1": [], "rouge-1_precision": [], "rouge-1_recall": []},
        "rouge-2": {"rouge-2_f1": [], "rouge-2_precision": [], "rouge-2_recall": []},
        "rouge-l": {"rouge-l_f1": [], "rouge-l_precision": [], "rouge-l_recall": []}
    }

    for text in target_texts:
        scores = rouge.get_scores(simplified, text, avg=True)
        rouge_scores["rouge-1"]["rouge-1_f1"].append(scores['rouge-1']['f'])
        rouge_scores["rouge-1"]["rouge-1_precision"].append(scores['rouge-1']['p'])
        rouge_scores["rouge-1"]["rouge-1_recall"].append(scores['rouge-1']['r'])

        rouge_scores["rouge-2"]["rouge-2_f1"].append(scores['rouge-2']['f'])
        rouge_scores["rouge-2"]["rouge-2_precision"].append(scores['rouge-2']['p'])
        rouge_scores["rouge-2"]["rouge-2_recall"].append(scores['rouge-2']['r'])

        rouge_scores["rouge-l"]["rouge-l_f1"].append(scores['rouge-l']['f'])
        rouge_scores["rouge-l"]["rouge-l_precision"].append(scores['rouge-l']['p'])
        rouge_scores["rouge-l"]["rouge-l_recall"].append(scores['rouge-l']['r'])

    return rouge_scores

# --------------------
# ------ SummaC ------
# --------------------
def calculate_summac(summac_zs_model, summac_conv_model, original, simplified):
    if not original.strip() or not simplified.strip():
        return 0.0, 0.0
    score_zs = summac_zs_model.score([original], [simplified])
    score_conv = summac_conv_model.score([original], [simplified])
    return score_zs["scores"][0], score_conv["scores"][0]

# ***************************************
# ****** EVALUACIÓN REESTRUCTURADA ******
# ***************************************
def evaluate_pair(name, original, simplified, references_text, st_model, idf_dict_ref, idf_dict_hyp, summac_zs_model, summac_conv_model, align_scorer, cfg):

    # ====== REFERENCIAS ======
    reference_ids = [f"ref_{i+1}" for i in range(len(references_text))]

    # ====== MÉTRICAS DEPENDIENTES DE REFERENCIA ======
    # --- SARI ---
    sari_add, sari_keep, sari_del, sari_total = [], [], [], []
    for ref in references_text:
        add, keep, delete, sari = calculate_sari(original, simplified, [ref])
        sari_add.append(round(add, 4))
        sari_keep.append(round(keep, 4))
        sari_del.append(round(delete, 4))
        sari_total.append(round(sari, 4))

    # ====== MÉTRICAS ORIGINAL vs SIMPLIFICADO ======
    # Se calcula UNA vez por par, no en bucle por referencia

    # --- BERTScore ---
    P, R, F1 = calculate_bertscore(simplified, original, cfg)
    bert_P, bert_R, bert_F1 = round(P, 4), round(R, 4), round(F1, 4)

    # --- SAS ---
    sas_val = round(SAS(simplified, original, st_model), 4)

    # --- MoverScore ---
    mover_val = round(calculate_moverscore(simplified, [original], idf_dict_ref, idf_dict_hyp, cfg)[0], 4)

    # --- BLEU ---
    b = calculate_bleu(simplified, [original])  
    bleu_vals = {k: round(v[0], 4) for k, v in b.items()}

    # --- ROUGE ---
    r = calculate_rouge(simplified, [original]) 
    rouge_vals = {
        "rouge-1_f1": round(r["rouge-1"]["rouge-1_f1"][0], 4),
        "rouge-1_precision": round(r["rouge-1"]["rouge-1_precision"][0], 4),
        "rouge-1_recall": round(r["rouge-1"]["rouge-1_recall"][0], 4),
        "rouge-2_f1": round(r["rouge-2"]["rouge-2_f1"][0], 4),
        "rouge-2_precision": round(r["rouge-2"]["rouge-2_precision"][0], 4),
        "rouge-2_recall": round(r["rouge-2"]["rouge-2_recall"][0], 4),
        "rouge-l_f1": round(r["rouge-l"]["rouge-l_f1"][0], 4),
        "rouge-l_precision": round(r["rouge-l"]["rouge-l_precision"][0], 4),
        "rouge-l_recall": round(r["rouge-l"]["rouge-l_recall"][0], 4)
    }

    # --- ALIGNScore ---
    alignscore = round(Align_Score(original, simplified, align_scorer), 4)

    # --- SummaC ---
    summac_zs, summac_conv = calculate_summac(summac_zs_model, summac_conv_model, original, simplified)

    # --- QuestEval ---
    q = Quest_Eval(original, simplified)
    quest_val = round(q["corpus_score"], 4)

    # --- Readability ---
    readability_original = calculate_readability(original, cfg)
    readability_refs = [calculate_readability(ref, cfg) for ref in references_text]
    readability_simplified = calculate_readability(simplified, cfg)

    # =============
    #      JSON 
    # =============
    
    result = {
        "id_original_text": name,
        "original_text": original,
        "id_reference_text": reference_ids,
        "reference_text": references_text,
        "simplified_text": simplified,
        "model_name": MODEL_NAME,
        "scores": {
            "simplification": {
                "sari": {
                    "sari_total": sari_total,
                    "sari_add": sari_add,
                    "sari_keep": sari_keep,
                    "sari_del": sari_del
                }
            },
            "similarity": {
                "bertscore": {
                    "bertscore_precision": bert_P,
                    "bertscore_recall": bert_R,
                    "bertscore_f1": bert_F1
                },
                "moverscore": mover_val,
                "sas": sas_val,
                "bleu": bleu_vals,
                "rouge": rouge_vals
            },
            "factuality": {
                "alignscore": alignscore,
                "summac": {
                    "summac_zs": round(summac_zs, 4),
                    "summac_conv": round(summac_conv, 4)
                },
                "questeval": quest_val
            },
            "readability": {
                "original": readability_original,
                "references": readability_refs,
                "simplified": readability_simplified
            }
        }
    }

    return [result]


# **************************
# ****** EXPORTAR CSV ******
# **************************
def results_to_csv(results, output_path):

    csv_rows = []

    for r in results:
        refs = r["id_reference_text"] 
        sari = r["scores"]["simplification"]["sari"]
        bert = r["scores"]["similarity"]["bertscore"]
        bleu = r["scores"]["similarity"]["bleu"]
        rouge = r["scores"]["similarity"]["rouge"]
        
        # Estas métricas ahora son de valor único
        mover = r["scores"]["similarity"]["moverscore"]
        sas_st = r["scores"]["similarity"]["sas"]
        quest = r["scores"]["factuality"]["questeval"]
        
        ro = r["scores"]["readability"]["original"]
        rs = r["scores"]["readability"]["simplified"]

        for i in range(len(refs)):
            rr = r["scores"]["readability"]["references"][i]

            row = {
                "id_original_text": r["id_original_text"],
                "id_reference_text": refs[i],
                "model_name": r["model_name"],

                # SARI (Dependiente de referencia)
                "sari_total": sari["sari_total"][i],
                "sari_add": sari["sari_add"][i],
                "sari_keep": sari["sari_keep"][i],
                "sari_del": sari["sari_del"][i],

                # Métricas Original vs Simplificado (se repite el valor para todas las referencias)
                "bertscore_precision": bert["bertscore_precision"],
                "bertscore_recall": bert["bertscore_recall"],
                "bertscore_f1": bert["bertscore_f1"],

                "moverscore": mover,
                "sas": sas_st,

                "bleu": bleu["bleu"],
                "bleu_precision_1": bleu["bleu_precision-1"],
                "bleu_precision_2": bleu["bleu_precision-2"],
                "bleu_precision_3": bleu["bleu_precision-3"],
                "bleu_precision_4": bleu["bleu_precision-4"],
                "bleu_brevity_penalty": bleu["bleu_brevity_penalty"],
                "bleu_length_ratio": bleu["bleu_length_ratio"],
                "bleu_translation_length": bleu["bleu_translation_length"],
                "bleu_reference_length": bleu["bleu_reference_length"],

                "rouge_1_f1": rouge["rouge-1_f1"],
                "rouge_1_precision": rouge["rouge-1_precision"],
                "rouge_1_recall": rouge["rouge-1_recall"],

                "rouge_2_f1": rouge["rouge-2_f1"],
                "rouge_2_precision": rouge["rouge-2_precision"],
                "rouge_2_recall": rouge["rouge-2_recall"],

                "rouge_l_f1": rouge["rouge-l_f1"],
                "rouge_l_precision": rouge["rouge-l_precision"],
                "rouge_l_recall": rouge["rouge-l_recall"],

                "alignscore": r["scores"]["factuality"]["alignscore"],
                "summac_zs": r["scores"]["factuality"]["summac"]["summac_zs"],
                "summac_conv": r["scores"]["factuality"]["summac"]["summac_conv"],
                "questeval": quest,

                # Readability
                "readability_original": list(ro.values())[0] if ro else None,
                "readability_reference": list(rr.values())[0] if rr else None,
                "readability_simplified": list(rs.values())[0] if rs else None,
            }

            csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    df.to_csv(output_path, index=False, sep=";", encoding="utf-8", decimal=",")

# **************************
# *** PARSEAR ARGUMENTOS ***
# **************************
def parse_args():
    parser = argparse.ArgumentParser(description="Bilingual metrics (ES/EN)")
    parser.add_argument("--lang", choices=["es", "en"], required=True,
                        help="Idioma de evaluación: es | en")
    parser.add_argument("--input_folder", required=True,
                        help="Carpeta con archivos JSON (nombre del archivo = modelo, estructura: {field: {original, simplified, reference}})")
    parser.add_argument("--out", required=True,
                        help="Carpeta de salida para JSON/CSV/CodeCarbon")
    return parser.parse_args()

# **********************
# ****** MAIN CSV ******
# **********************
if __name__ == "__main__":

    args = parse_args()
    cfg = LANG_CFG[args.lang]

    # Input folder with JSON files (one per model)
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        raise ValueError(f"❌ Input folder does not exist: {input_folder}")

    OUTPUT_FOLDER = args.out
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get all JSON files from input folder (model files)
    model_files = sorted([f for f in input_folder.glob("*.json")])
    
    if not model_files:
        raise ValueError(f"❌ No JSON files found in {input_folder}")

    print(f"📁 Archivos JSON detectados en {input_folder}: {[f.stem for f in model_files]}\n")
    print(f"📂 Resultados se guardarán en: {OUTPUT_FOLDER}\n")

    # Initialize sentence transformer
    st_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    # Initialize SummaC and AlignScore
    model_zs_summac, model_conv_summac = init_summac(cfg)
    align_scorer = init_alignscore(cfg, device="cuda:0", batch_size=32)

    # Global emissions tracker
    emissions_path_global = os.path.join(OUTPUT_FOLDER, "emissions_metricas_GLOBAL.csv")
    tracker_global = OfflineEmissionsTracker(
        project_name="metricas_GLOBAL",
        output_file=emissions_path_global,
        country_iso_code="ESP",
    )

    # Global timer
    total_start = time.time()
    tracker_global.start()

    try:    
        # Process each model file
        for model_file in tqdm(model_files, desc="Procesando modelos", unit="modelo", position=0, leave=True):
            start_time = time.time()
            
            MODEL_NAME = model_file.stem
            
            print("\n" + "="*90)
            print(f"🚀 Procesando modelo: {MODEL_NAME}")
            print("="*90)

            emissions_path_model = os.path.join(OUTPUT_FOLDER, f"emissions_metricas_{MODEL_NAME}.csv")        
            output_path_json = os.path.join(OUTPUT_FOLDER, f"metrics_{MODEL_NAME}.json")
            output_path_csv = os.path.join(OUTPUT_FOLDER, f"metrics_{MODEL_NAME}.csv")
            
            # Per-model emissions tracker
            tracker_model = OfflineEmissionsTracker(
                project_name=f"metricas_{MODEL_NAME}",
                output_file=emissions_path_model,
                country_iso_code="ESP",
            )

            tracker_model.start()
            all_results = []

            try:
                # Load model data from JSON file
                with open(model_file, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                
                print(f"Pares encontrados: {len(model_data)}")

                # Extract all texts for IDF calculation
                all_originals = []
                all_simplified = []
                
                for field_name, content in model_data.items():
                    if isinstance(content, dict):
                        original = content.get('original', '')
                        simplified = content.get('simplified', '')
                        
                        if original:
                            all_originals.append(original)
                        if simplified:
                            all_simplified.append(simplified)

                # Calculate global IDF dictionaries
                idf_dict_ref = get_idf_dict(all_originals)
                idf_dict_hyp = get_idf_dict(all_simplified)

                # Process each pair in the JSON
                for field_name, content in tqdm(sorted(model_data.items()), desc=f"Evaluando {MODEL_NAME}", unit="caso", position=1, leave=False):
                    if not isinstance(content, dict):
                        continue
                    
                    original = content.get('original', '')
                    simplified = content.get('simplified', '')
                    reference = content.get('reference', '')
                    
                    # Validar campos vacíos para evitar crasheos en BERTScore/ROUGE
                    # if not original.strip() or not simplified.strip() or not reference.strip():
                    #     print(f"\n⚠️ Saltando {field_name}: contiene campos vacíos")
                    #     continue
                    
                    # Handle reference as either single string or list
                    references = [reference] if isinstance(reference, str) else reference
                    references = references[:5] if len(references) > 5 else references
                    
                    # Evaluate pair
                    pair_results = evaluate_pair(
                        field_name, original, simplified, references,
                        st_model, idf_dict_ref, idf_dict_hyp,
                        model_zs_summac, model_conv_summac, align_scorer, cfg
                    )
                    all_results.extend(pair_results)
                    
                # Sort results by ID
                all_results = sorted(all_results, key=lambda r: r.get('id_original_text', ''))

                # Save results to JSON
                with open(output_path_json, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

                # Save results to CSV
                results_to_csv(all_results, output_path_csv)

            finally:
                tracker_model.stop()

                # Normalize CodeCarbon CSV (model) to European format
                if os.path.exists(emissions_path_model):
                    df_em = pd.read_csv(emissions_path_model)
                    df_em.to_csv(emissions_path_model, sep=";", decimal=",", index=False, encoding="utf-8")
                    print(f"✅ Emisiones ({MODEL_NAME}) guardadas (formato europeo): {emissions_path_model}")

            # Time per model
            elapsed = time.time() - start_time
            mins, secs = divmod(elapsed, 60)
            print(f"✅ Resultados guardados en {output_path_json}")
            print(f"✅ Resultados guardados en {output_path_csv}")
            print(f"📊 Total de evaluaciones: {len(all_results)}")
            print(f"⏱️ Tiempo de ejecución para {MODEL_NAME}: {int(mins)} min {int(secs)} s\n")
    
    finally:
        tracker_global.stop()

        # Normalize global CodeCarbon CSV to European format
        if os.path.exists(emissions_path_global):
            df_em = pd.read_csv(emissions_path_global)
            df_em.to_csv(emissions_path_global, sep=";", decimal=",", index=False, encoding="utf-8")
            print(f"✅ Emisiones GLOBAL guardadas (formato europeo): {emissions_path_global}")

        # Total time
        total_elapsed = time.time() - total_start
        total_mins, total_secs = divmod(total_elapsed, 60)
        total_hours, total_mins = divmod(total_mins, 60)
        print("=" * 90)
        print(f"🏁 Proceso completo finalizado.")
        print(f"🕒 Tiempo total: {int(total_hours)} h {int(total_mins)} min {int(total_secs)} s")
        print("=" * 90)