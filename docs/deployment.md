# Deployment

Home server, Docker Compose, image published to GHCR by CI
(`build-and-publish.yaml`, same as koebako/koemichi/tana).

## docker-compose.yaml (target shape)

```yaml
name: omozo

services:
  omozo:
    image: ghcr.io/divin/omozo:latest
    user: "${HOST_UID:-1000}:${HOST_GID:-1000}"
    ports:
      - "${OMOZO_PORT:-9095}:8000"
    environment:
      OMOZO_EMBED_URL: ${OMOZO_EMBED_URL:?Set OMOZO_EMBED_URL in .env}
      OMOZO_EMBED_MODEL: ${OMOZO_EMBED_MODEL:-bge-m3}
      OMOZO_RERANK_URL: ${OMOZO_RERANK_URL:-}
      OMOZO_RERANK_MODEL: ${OMOZO_RERANK_MODEL:-bge-reranker-v2-m3}
      OMOZO_MCP_TOKEN: ${OMOZO_MCP_TOKEN:-}
      OMOZO_PREVIEW_CHARS: ${OMOZO_PREVIEW_CHARS:-600}
      OMOZO_DEDUPE_THRESHOLD: ${OMOZO_DEDUPE_THRESHOLD:-0.92}
      OMOZO_MEMORY_MIN_SIM: ${OMOZO_MEMORY_MIN_SIM:-0.35}
      OMOZO_DOC_MIN_SIM: ${OMOZO_DOC_MIN_SIM:-0.35}
      OMOZO_RELATIVE_GAP: ${OMOZO_RELATIVE_GAP:-0.75}
      OMOZO_DEBUG_SCORES: ${OMOZO_DEBUG_SCORES:-}
      DATA_DIR: /data
      OMOZO_DOCS_DIRS: ${OMOZO_DOCS_DIRS:-/data/docs-1}
      TZ: ${TZ:-Europe/Berlin}
    volumes:
      - ${HOST_DATA_DIR:-./data}:/data
      - ${HOST_DOCS_1:?Set HOST_DOCS_1 in .env}:/data/docs-1:ro
      # Additional ingest roots: add mounts + extend OMOZO_DOCS_DIRS.
      # - ${HOST_DOCS_2}:/data/docs-2:ro
    restart: unless-stopped

  # Optional: scheduled ingestion. Comment out to ingest only by hand.
  ingest-worker:
    image: ghcr.io/divin/omozo:latest
    user: "${HOST_UID:-1000}:${HOST_GID:-1000}"
    command: ["omozo", "ingest-loop", "--interval", "3600"]
    environment:
      OMOZO_EMBED_URL: ${OMOZO_EMBED_URL:?Set OMOZO_EMBED_URL in .env}
      OMOZO_EMBED_MODEL: ${OMOZO_EMBED_MODEL:-bge-m3}
      OMOZO_DOCS_DIRS: ${OMOZO_DOCS_DIRS:-/data/docs-1}
      DATA_DIR: /data
      TZ: ${TZ:-Europe/Berlin}
    volumes:
      - ${HOST_DATA_DIR:-./data}:/data
      - ${HOST_DOCS_1:?Set HOST_DOCS_1 in .env}:/data/docs-1:ro
    restart: unless-stopped
```

Port 9095 is a placeholder — change `OMOZO_PORT` in `.env` without touching
code.

## example.env

```sh
# llama-server embedding endpoint (OpenAI-compatible /v1/embeddings).
OMOZO_EMBED_URL=http://host.docker.internal:8081
OMOZO_EMBED_MODEL=bge-m3

# Rerank endpoint; leave unset to disable reranking.
OMOZO_RERANK_URL=http://host.docker.internal:8082
OMOZO_RERANK_MODEL=bge-reranker-v2-m3

# Optional bearer token for the MCP endpoint; leave unset for trusted LAN use.
# OMOZO_MCP_TOKEN=replace-with-a-long-random-token

# Host paths. Docs roots are mounted read-only; one volume per root.
HOST_DATA_DIR=./data
HOST_DOCS_1=/srv/books
# HOST_DOCS_2=/srv/notes
# Colon-separated container paths matching the mounts above; used by
# `omozo ingest --all`.
OMOZO_DOCS_DIRS=/data/docs-1
OMOZO_PORT=9095
TZ=Europe/Berlin
```

## CLI

Run admin commands in the container:

```sh
docker compose exec omozo omozo ingest --all
docker compose exec omozo omozo ingest /data/docs-1/mybook.epub
docker compose exec omozo omozo ingest /data/docs-2/notes --prune
docker compose exec omozo omozo stats
docker compose exec omozo omozo gc --age-days 30  # purge superseded beyond age floor
docker compose exec omozo omozo forget <memory-id>
docker compose exec omozo omozo transcripts "fog signal"  # FTS over stored transcripts
```

`omozo serve` (the container default command) runs the MCP server.

## Client configuration

Point the local agent's MCP client at:

```
http://<server>:9095/mcp
```

with the bearer token if `OMOZO_MCP_TOKEN` is set.

## Startup validation

On boot the server embeds a probe string against `OMOZO_EMBED_URL` and
compares the dimension with `meta` in the DB. Mismatch or unreachable
endpoint → abort with a clear message. When the embedding model is ever
changed, re-embed everything (`omozo ingest --reembed` plus a memory
re-embedding pass) — Omozo refuses to run mixed-dimension data.

## Hugging Face tokenizer cache

First ingestion downloads the bge-m3 tokenizer (a few MB) into the
container's HF cache — which vanishes on container recreation unless
persisted via `HF_HOME: /data/hf` (set in the compose file).

## OCR models

Scanned PDFs are OCR'd with RapidOCR; its models (~30 MB: detection,
angle classification, recognition) download on the first scanned-document
conversion into `${DATA_DIR}/models/rapidocr` — inside the persisted
`/data` volume — and are reused afterwards. They are explicitly pinned via
`OMOZO_MODELS_DIR` because rapidocr's own auto-download targets the
read-only site-packages directory. Override the location with
`OMOZO_MODELS_DIR` if needed. Ingest failures are per-file: a broken
document is logged (see `IngestReport.failed` in the ingest-loop output)
and the run continues; the file is retried on the next tick.
