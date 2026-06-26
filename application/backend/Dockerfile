FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ make cmake \
    build-essential python3-dev \
    libffi-dev libssl-dev \
    libblas-dev liblapack-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --- NEW: Copy BOTH local dependencies before pip install ---
COPY summac/ ./summac/
COPY QuestEval/ ./QuestEval/

# Now pip install will succeed
RUN pip install -r requirements.txt hf-transfer

RUN python3 -m spacy download es_core_news_md
RUN python3 -m spacy download en_core_web_md

ARG HF_TOKEN

RUN HF_TOKEN=${HF_TOKEN} HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_VERBOSITY=debug python3 -u -c "\
import os; \
from huggingface_hub import snapshot_download; \
print('🚀 Starting high-speed model downloads with DEBUG logs...'); \
print('⏳ Downloading HiTZ/Latxa-Qwen3-VL-8B-Instruct...'); \
snapshot_download(repo_id='HiTZ/Latxa-Qwen3-VL-8B-Instruct'); \
print('⏳ Downloading IIC/RigoChat-7b-v2...'); \
snapshot_download(repo_id='IIC/RigoChat-7b-v2'); \
print('✅ All downloads complete!')"

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

EXPOSE 3232

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:3232", "--timeout", "1000", "wsgi:app"]