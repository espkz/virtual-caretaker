#!/usr/bin/env bash
set -euo pipefail

APP_NAME="virtual-caretaker"
PORT_MAP="${PORT_MAP:-8080:8080}"
SQLITE_VOLUME="${SQLITE_VOLUME:-vc_sqlite_data}"

podman stop "$APP_NAME" || true
podman rm "$APP_NAME" || true
podman image rm "$APP_NAME" || true

podman build -t "$APP_NAME" .

podman run -d \
  --name "$APP_NAME" \
  --env-file .env \
  --restart=always \
  -p "$PORT_MAP" \
  -v "$SQLITE_VOLUME":/app/data \
  "$APP_NAME"

podman logs --tail 50 "$APP_NAME"
