from pathlib import Path
from bert_score import score

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
        "alignscore_ckpt_path": None, 
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

# -----------------------
# ------ BERTScore ------
# -----------------------
def calculate_bertscore(simplified, target_text, cfg):
    if not simplified.strip() or not target_text.strip():
        return 0.0, 0.0, 0.0
    P, R, F1 = score(
        [simplified], 
        [target_text], 
        model_type=cfg["bertscore_model"], 
        num_layers=12, 
        lang=cfg["bertscore_lang"], 
        rescale_with_baseline=False
    )
    return P.mean().item(), R.mean().item(), F1.mean().item()