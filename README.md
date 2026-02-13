# HWPULSE

Real-time terminal system monitor (CPU/GPU/RAM)

<img width="1193" height="714" alt="demo" src="https://github.com/user-attachments/assets/dec808c1-dfb8-43fc-9f74-ddbf81c719e1" />

## Features

- Color thresholds for load, temperature, and power.
- Min/Max history for each metric since startup.
- Graphs for key metrics (default mode).
- JSON modes for integration with external programs.

## Requirements

- Linux
- Python **3.9+** (3.10+ recommended)
- `free` (from `procps` package)
- For GPU metrics: `nvidia-smi`
- For CPU temperature: `sensors` (from `lm-sensors` package)
- For CPU power: `/sys/class/powercap/intel-rapl:0/energy_uj` (if missing, the metric is unavailable)

## Install Dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3 procps util-linux lm-sensors
```

Optional sensor initialization:

```bash
sudo sensors-detect --auto
```

For GPU metrics, NVIDIA driver must be installed (`nvidia-smi` should work).

## Files

- `hwpulse.py` - main script.
- `install.sh` - installs `hwpulse` (copies to `/opt/hwpulse` + launcher in `/usr/local/bin`).
- `uninstall.sh` - removes launcher from `/usr/local/bin` and script from `/opt/hwpulse`.

## Run Modes

1. UI with graphs (default):

```bash
python3 hwpulse.py
```

2. UI without graphs:

```bash
python3 hwpulse.py --nograph
```

3. JSON over stdin/stdout (for IPC):

```bash
python3 hwpulse.py --json-stdio
```

- Process runs continuously.
- Request command: `get` (line `get\n` to stdin).
- Each `get` returns one JSON object to stdout.

4. One-shot JSON snapshot:

```bash
python3 hwpulse.py --json-once
```

- Script collects metrics once, prints JSON, and exits.

Help:

```bash
python3 hwpulse.py --help
```

## JSON Format

- Root fields:
  - `timestamp` - Unix time in milliseconds.
  - `systemName` - system name (from `PRETTY_NAME`).
  - `kernel` - kernel version (from `uname -r`).
  - `cpu` - CPU object (always present).
  - `gpu` - GPU object (present only if GPU is detected).

- Inside `cpu`/`gpu`:
  - `modelName`
  - available metrics for that device.

Population rules:

- If a metric is not supported on the system, its key is omitted from JSON.
- If a metric is supported but failed to read at a specific moment, its value is `null`.

Example:

```json
{
  "timestamp": 1770978423351,
  "systemName": "Debian GNU/Linux 12 (bookworm)",
  "kernel": "6.12.69+deb13-amd64",
  "cpu": {
    "modelName": "AMD Ryzen 9 9950X 16-Core Processor",
    "loadPercent": 0.0,
    "tempC": 49.8,
    "powerW": 6.8,
    "freqAvgMhz": 3131.0,
    "freqMaxMhz": 5062.0,
    "ramUsedMb": 3310.0,
    "ramTotalMb": 61835.0
  },
  "gpu": {
    "modelName": "NVIDIA GeForce RTX 5090",
    "loadPercent": 0.0,
    "tempC": 43.0,
    "powerW": 33.02,
    "fanPercent": 0.0,
    "coreMhz": 240.0,
    "memMhz": 405.0,
    "vramUsedMb": 147.0,
    "vramTotalMb": 32607.0
  }
}
```

## Global `hwpulse` Command

To enable `hwpulse` from any directory:

```bash
chmod +x install.sh
./install.sh
```

After that, you can run:

```bash
hwpulse
hwpulse --nograph
hwpulse --json-stdio
hwpulse --json-once
```

## Uninstall

```bash
chmod +x uninstall.sh
./uninstall.sh
```
