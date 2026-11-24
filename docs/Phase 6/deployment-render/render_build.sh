#!/usr/bin/env bash
set -euo pipefail

OWNER="${MODEL_REPO_OWNER:?MODEL_REPO_OWNER is not set}"
REPO="${MODEL_REPO_NAME:-aida-project}"
TAG="${MODEL_TAG:-model-v2026-04-25}"
ASSET_NAME="catboost_price.cbm"

: "${GITHUB_TOKEN:?GITHUB_TOKEN is not set}"
echo "Using tag: ${TAG}"
echo "Looking for asset: ${ASSET_NAME}"

mkdir -p data/models

# 1) Fetch release JSON and capture HTTP status
HTTP_CODE=$(
  curl -sS -L \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -o /tmp/release.json \
    -w "%{http_code}" \
    "https://api.github.com/repos/${OWNER}/${REPO}/releases/tags/${TAG}"
)

echo "GitHub release API status: ${HTTP_CODE}"

if [ "${HTTP_CODE}" != "200" ]; then
  echo "ERROR: release API did not return 200. Response (first 400 chars):"
  head -c 400 /tmp/release.json || true
  echo
  exit 1
fi

# 2) Extract asset_id from JSON
ASSET_ID=$(
python - <<'PY'
import json
path = "/tmp/release.json"
asset_name = "catboost_price.cbm"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for a in data.get("assets", []):
    if a.get("name") == asset_name:
        print(a["id"])
        raise SystemExit(0)

raise SystemExit(f"Asset not found in release: {asset_name}")
PY
)

echo "Asset id: ${ASSET_ID}"

# 3) Download the asset binary (use octet-stream for asset download)
DL_CODE=$(
  curl -sS -L \
    -H "Accept: application/octet-stream" \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -o "data/models/${ASSET_NAME}" \
    -w "%{http_code}" \
    "https://api.github.com/repos/${OWNER}/${REPO}/releases/assets/${ASSET_ID}"
)

echo "GitHub asset download status: ${DL_CODE}"

if [ "${DL_CODE}" != "200" ]; then
  echo "ERROR: asset download did not return 200."
  exit 1
fi

ls -lh "data/models/${ASSET_NAME}"

# 4) Install dependencies using the same Python interpreter
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -c "import requests; print('requests ok')"