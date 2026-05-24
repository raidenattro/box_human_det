#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose build visual-dps-event-worker
echo "OK: visual-dps-event-worker:latest"
