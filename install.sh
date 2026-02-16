#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
APP_DIR="/opt/hwpulse"
TARGET="/usr/local/bin/hwpulse"

shopt -s nullglob
SOURCE_FILES=("$SCRIPT_DIR"/hwpulse*.py)
shopt -u nullglob

if [[ ${#SOURCE_FILES[@]} -eq 0 ]]; then
  echo "Error: no hwpulse*.py files found in: $SCRIPT_DIR" >&2
  exit 1
fi

sudo install -d -m 755 "$APP_DIR"
sudo find "$APP_DIR" -maxdepth 1 -type f -name 'hwpulse*.py' -delete

for src in "${SOURCE_FILES[@]}"; do
  dst="$APP_DIR/$(basename "$src")"
  sudo install -m 644 "$src" "$dst"
done

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
echo "Installed scripts:"
for src in "${SOURCE_FILES[@]}"; do
  echo "  $(basename "$src")"
done
echo "Run: hwpulse"