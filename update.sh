#!/usr/bin/env bash
set -euo pipefail

APP_NAME="virtual-caretaker"
BRANCH="${BRANCH:-main}"
PORT_MAP="${PORT_MAP:-8080:8080}"
SQLITE_VOLUME="${SQLITE_VOLUME:-vc_sqlite_data}"

cd "$(dirname "$0")"

echo "[1/6] Updating code..."
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "[2/6] Ensuring sqlite volume exists..."
podman volume create "$SQLITE_VOLUME" >/dev/null || true

echo "[3/6] Stopping old container/image..."
podman stop "$APP_NAME" || true
podman rm "$APP_NAME" || true
podman image rm "$APP_NAME" || true

echo "[4/6] Building image..."
podman build -t "$APP_NAME" .

echo "[5/6] Starting container..."
podman run -d \
  --name "$APP_NAME" \
  --env-file .env \
  --restart=always \
  -p "$PORT_MAP" \
  -v "$SQLITE_VOLUME":/app/data \
  "$APP_NAME"

echo "[6/6] Done. Recent logs:"
podman logs --tail 50 "$APP_NAME"

