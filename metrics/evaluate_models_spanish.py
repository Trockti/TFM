from collections import defaultdict
import os
import sys
from pathlib import Path


from much.src.annotation import get_all_annotations
from much.src.utils.constants import AUTHORIZED_LANG

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, cohen_kappa_score
from matplotlib import pyplot as plt
from collections import defaultdict

LLMS_INFOS = {
    "meta-llama/Llama-3.2-3B-Instruct": dict(filter_date=True, pad_tok_id=128011),
    "HiTZ/Latxa-Qwen3-VL-8B-Instruct": dict(filter_date=False, pad_tok_id=128011),
    "mistralai/Ministral-3-8B-Instruct-2512": dict(filter_date=False, pad_tok_id=11),
    "IIC/RigoChat-7b-v2": dict(filter_date=False, pad_tok_id=6),
}


stats = {model_name:{'-1':0, "total":0} for model_name in ["total"] + list(LLMS_INFOS)}
all_annotations = get_all_annotations(reload=True)
for _, annotation in all_annotations.items():
    if "gpt-gemma4" not in annotation.labels or -1 not in annotation.labels['gpt-gemma4']:
        continue
    model_name = annotation.generation.generation_cfg.model_name
    # Only process models that are in LLMS_INFOS
    if model_name not in LLMS_INFOS:
        continue
    stats["total"]['-1'] += np.unique(annotation.labels['gpt-gemma4'][:-1], return_counts=True)[1][0].item()
    stats["total"]['total'] += len(annotation.labels['gpt-gemma4'][:-1])
    stats[model_name]['-1'] += np.unique(annotation.labels['gpt-gemma4'][:-1], return_counts=True)[1][0].item()
    stats[model_name]['total'] += len(annotation.labels['gpt-gemma4'][:-1])

for model_name, stat in stats.items():
    print(f"Model: {model_name} | -1: {stat['-1']} | Total: {stat['total']} | % -1: {stat['-1']/stat['total']:.2%}")