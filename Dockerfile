# Jobava London Rationale Explorer — container image.
#
# Build:  docker build -t jobava-explorer .
# Run:    docker run -p 8501:8501 jobava-explorer
#         (optional) -e ANTHROPIC_API_KEY=sk-ant-... for AI explanations
#
# Works as-is on container hosts such as Render, Railway, Fly.io and
# Google Cloud Run (they set $PORT; the CMD below respects it).

FROM python:3.12-slim

# Stockfish from apt so engine analysis works out of the box.
RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*
ENV STOCKFISH_PATH=/usr/games/stockfish

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
