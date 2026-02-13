#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_SCRIPT="$SCRIPT_DIR/hwpulse.py"
APP_DIR="/opt/hwpulse"
INSTALLED_SCRIPT="$APP_DIR/hwpulse.py"
TARGET="/usr/local/bin/hwpulse"

if [[ ! -f "$SOURCE_SCRIPT" ]]; then
  echo "Error: script not found: $SOURCE_SCRIPT" >&2
  exit 1
fi

sudo install -d -m 755 "$APP_DIR"
sudo install -m 755 "$SOURCE_SCRIPT" "$INSTALLED_SCRIPT"

sudo tee "$TARGET" > /dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/opt/hwpulse/hwpulse.py"

if [[ $EUID -ne 0 ]]; then
  exec sudo python3 "$SCRIPT" "$@"
else
  exec python3 "$SCRIPT" "$@"
fi
EOF

sudo chmod +x "$TARGET"
echo "Installed: $TARGET"
echo "Installed script: $INSTALLED_SCRIPT"
echo "Run: hwpulse"

