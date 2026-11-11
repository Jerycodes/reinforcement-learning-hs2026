FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY kart_rl ./kart_rl
RUN pip install --no-cache-dir . && useradd --system --create-home racer && mkdir -p runs/arcade && chown racer runs/arcade
COPY viewer ./viewer
USER racer
CMD ["python","-m","kart_rl.game","--arcade","--lan","--http","8031","--ws","8796"]
