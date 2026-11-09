# Servizio "posa" per i negozi Shopify (API + agenti + render). Niente Gradio: Pillow >= 13.
ARG BASE=python:3.12-slim
FROM ${BASE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/modelli POSA_DATI=/dati

# nessun pacchetto di sistema: OpenCV "headless" non usa librerie grafiche
RUN useradd --create-home --uid 10001 posa \
 && mkdir -p /dati /modelli && chown posa:posa /dati /modelli

WORKDIR /srv/posa
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install ".[server,segmentazione]" \
 && python -c "import PIL; assert int(PIL.__version__.split('.')[0]) >= 13, PIL.__version__"

USER posa
# scarica il modello di segmentazione (versione fissata, safetensors) dentro l'immagine
RUN posa verifica --modelli || true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/salute', timeout=4)"
CMD ["posa", "server", "--host", "0.0.0.0", "--porta", "8000"]
