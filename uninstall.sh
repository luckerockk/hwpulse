#!/usr/bin/env bash
set -euo pipefail

TARGET_BIN="/usr/local/bin/hwpulse"
APP_DIR="/opt/hwpulse"

if [[ -e "$TARGET_BIN" ]]; then
  sudo rm -f "$TARGET_BIN"
  echo "Removed: $TARGET_BIN"
else
  echo "Not found: $TARGET_BIN"
fi

if [[ -d "$APP_DIR" ]]; then
  shopt -s nullglob
  INSTALLED_FILES=("$APP_DIR"/hwpulse*.py)
  PYC_FILES=("$APP_DIR"/*.pyc)
  shopt -u nullglob

  if [[ ${#INSTALLED_FILES[@]} -gt 0 ]]; then
    for file in "${INSTALLED_FILES[@]}"; do
      sudo rm -f "$file"
      echo "Removed: $file"
    done
  else
    echo "Not found: $APP_DIR/hwpulse*.py"
  fi

  if [[ ${#PYC_FILES[@]} -gt 0 ]]; then
    for file in "${PYC_FILES[@]}"; do
      sudo rm -f "$file"
      echo "Removed: $file"
    done
  fi

  # Python bytecode cache dirs created at runtime.
  if sudo find "$APP_DIR" -type d -name '__pycache__' -print -exec rm -rf {} + | sed 's/^/Removed: /'; then
    :
  fi

  if sudo rmdir "$APP_DIR" 2>/dev/null; then
    echo "Removed empty dir: $APP_DIR"
  else
    echo "Kept dir (not empty): $APP_DIR"
  fi
else
  echo "Not found: $APP_DIR"
fi