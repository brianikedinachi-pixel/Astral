# Astral server — Docker deploy path for Render.
#
# Why this exists: Render's default "native Python" service just runs
# `pip install -r requirements.txt` — it can't apt-get anything. Kokoro
# (the self-hosted TTS engine, see Server.py's KOKORO_ENABLED) needs the
# espeak-ng system package for phonemizing out-of-vocabulary words. This
# Dockerfile is the only way to get that onto Render.
#
# Switching your Render service to this: in the Render dashboard, on this
# service → Settings → Build & Deploy → Runtime, change it from "Python 3"
# to "Docker", and point it at this Dockerfile (repo root, or wherever you
# place it — adjust the path in Render's settings to match).
#
# The two Kokoro model files (~350MB total) are downloaded once here, at
# build time — not at runtime — so every deploy ships with them already
# baked into the image and Server.py's warmup never has to download
# anything. If you'd rather not bloat the image, delete the two `curl`
# lines below and Server.py will download them itself on first boot
# instead (see _ensure_kokoro_model_files in Server.py) — just know that
# adds a one-time ~350MB download after every cold start on Render's free
# tier, since the filesystem there doesn't persist across restarts unless
# you've attached a paid persistent disk (RENDER_PERSISTENT_DIR).

FROM python:3.11-slim

# espeak-ng: required by Kokoro's phonemizer (misaki) for words outside its
# built-in lexicon. curl: only needed to fetch the model files below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the Kokoro model files into the image (matches the path Server.py's
# _KOKORO_MODEL_DIR resolves to when no persistent disk is attached: a
# "kokoro_model" folder next to Server.py — see SCRIPT_DIR in Server.py).
RUN mkdir -p scripts/kokoro_model \
    && curl -L -o scripts/kokoro_model/kokoro-v1.0.onnx \
        https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx \
    && curl -L -o scripts/kokoro_model/voices-v1.0.bin \
        https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

CMD ["python", "scripts/Server.py"]
