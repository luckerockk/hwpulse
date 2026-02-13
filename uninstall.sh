#!/usr/bin/env bash
set -euo pipefail

TARGET_BIN="/usr/local/bin/hwpulse"
APP_DIR="/opt/hwpulse"
INSTALLED_SCRIPT="$APP_DIR/hwpulse.py"

if [[ -e "$TARGET_BIN" ]]; then
  sudo rm -f "$TARGET_BIN"
  echo "Removed: $TARGET_BIN"
else
  echo "Not found: $TARGET_BIN"
fi

if [[ -e "$INSTALLED_SCRIPT" ]]; then
  sudo rm -f "$INSTALLED_SCRIPT"
  echo "Removed: $INSTALLED_SCRIPT"
else
  echo "Not found: $INSTALLED_SCRIPT"
fi

if [[ -d "$APP_DIR" ]]; then
  if sudo rmdir "$APP_DIR" 2>/dev/null; then
    echo "Removed empty dir: $APP_DIR"
  else
    echo "Kept dir (not empty): $APP_DIR"
  fi
fi

