#!/usr/bin/env python3
import math
import os
from collections import deque
from typing import Optional, Tuple

from hwpulse_common import (
    CPU_HWMON_CHIPS,
    CPU_RAPL,
    CPU_THERMAL_TYPES,
    GRAPH_POINTS,
    HWMON_BASE,
    SPARK_CHARS,
    THERMAL_BASE,
    run_cmd,
    strip_if_na,
    to_float_or_none,
)

def get_cpu_model() -> str:
    lscpu_out = run_cmd(["lscpu"])
    for line in lscpu_out.splitlines():
        if line.startswith("Model name:"):
            return strip_if_na(line.split(":", 1)[1])

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("model name"):
                    return strip_if_na(line.split(":", 1)[1])
    except Exception:
        pass

    return "N/A"

def get_system_info() -> Tuple[str, str, str]:
    pretty = ""
    try:
        with open("/etc/os-release", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    pretty = line.split("=", 1)[1].strip().strip('"')
                    break
    except Exception:
        pass

    host = run_cmd(["hostname"]).splitlines()
    hostname = host[0].strip() if host else ""

    kernel = run_cmd(["uname", "-r"]).splitlines()
    kernel_ver = kernel[0].strip() if kernel else ""

    return strip_if_na(pretty), strip_if_na(hostname), strip_if_na(kernel_ver)

def get_system_name() -> str:
    pretty, hostname, kernel_ver = get_system_info()

    parts: list[str] = []
    if pretty != "N/A":
        parts.append(pretty)
    if hostname != "N/A":
        parts.append(f"host {hostname}")
    if kernel_ver != "N/A":
        parts.append(f"kernel {kernel_ver}")

    if not parts:
        return "N/A"
    return " | ".join(parts)

def read_cpu_stat() -> Tuple[int, int]:
    try:
        with open("/proc/stat", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = line.split()
                    vals = [int(x) for x in parts[1:]]
                    idle = vals[3] + vals[4]
                    total = sum(vals)
                    return total, idle
    except Exception:
        return 0, 0
    return 0, 0

def read_cpu_energy() -> Optional[int]:
    try:
        with open(CPU_RAPL, "r", encoding="utf-8", errors="ignore") as f:
            return int(f.read().strip())
    except Exception:
        return None

def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""

def list_indexed_entries(base_dir: str, prefix: str) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    try:
        names = os.listdir(base_dir)
    except Exception:
        return items

    for name in names:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if not suffix.isdigit():
            continue
        items.append((int(suffix), name))

    items.sort(key=lambda x: x[0])
    return items

def parse_temp_c(raw_value: str) -> Optional[float]:
    raw = raw_value.strip()
    if not raw:
        return None

    try:
        value = float(raw)
    except ValueError:
        return None

    # thermal_zone*/temp and temp*_input are typically in millidegrees C.
    if abs(value) >= 1000.0:
        value /= 1000.0

    if value < -50.0 or value > 250.0:
        return None
    return value

def format_temp_c(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}"

def detect_cpu_temp_from_thermal() -> Optional[dict[str, object]]:
    temp_paths: list[str] = []

    for _, zone_name in list_indexed_entries(THERMAL_BASE, "thermal_zone"):
        base = os.path.join(THERMAL_BASE, zone_name)
        zone_type = read_text_file(os.path.join(base, "type")).strip().lower()
        if zone_type not in CPU_THERMAL_TYPES:
            continue

        temp_path = os.path.join(base, "temp")
        if not os.path.isfile(temp_path):
            continue

        temp_paths.append(temp_path)

    if not temp_paths:
        return None

    return {
        "kind": "thermal",
        "temp_paths": temp_paths,
    }

def hwmon_temp_path_priority(chip_name: str, label: str) -> int:
    chip = chip_name.lower()
    text = label.lower()

    if chip == "coretemp":
        if "package" in text:
            return 0
        if "tctl" in text or "tdie" in text:
            return 1
        if "core" in text:
            return 3
        return 5

    if chip == "k10temp":
        if "tdie" in text:
            return 0
        if "tctl" in text:
            return 1
        if "package" in text:
            return 2
        return 5

    return 10

def detect_cpu_temp_from_hwmon() -> Optional[dict[str, object]]:
    selected_paths: list[str] = []

    for _, hwmon_name in list_indexed_entries(HWMON_BASE, "hwmon"):
        base = os.path.join(HWMON_BASE, hwmon_name)
        chip_name = read_text_file(os.path.join(base, "name")).strip().lower()
        if chip_name not in CPU_HWMON_CHIPS:
            continue

        chip_paths: list[tuple[int, str]] = []
        try:
            entries = os.listdir(base)
        except Exception:
            continue

        temp_entries: list[tuple[int, str]] = []
        for entry in entries:
            if not entry.startswith("temp") or not entry.endswith("_input"):
                continue
            idx_text = entry[len("temp") : -len("_input")]
            if not idx_text.isdigit():
                continue
            temp_entries.append((int(idx_text), entry))

        temp_entries.sort(key=lambda x: x[0])

        for idx, entry in temp_entries:
            temp_path = os.path.join(base, entry)
            if not os.path.isfile(temp_path):
                continue

            label = read_text_file(os.path.join(base, f"temp{idx}_label"))
            priority = hwmon_temp_path_priority(chip_name, label)
            chip_paths.append((priority, temp_path))

        if not chip_paths:
            continue

        chip_paths.sort(key=lambda x: x[0])
        best_priority = chip_paths[0][0]
        preferred = [path for prio, path in chip_paths if prio == best_priority and prio < 5]
        selected = preferred if preferred else [path for _, path in chip_paths]
        selected_paths.extend(selected)

    if not selected_paths:
        return None

    return {
        "kind": "hwmon",
        "temp_paths": selected_paths,
    }

def detect_cpu_temp_source() -> Optional[dict[str, object]]:
    source = detect_cpu_temp_from_thermal()
    if source is not None:
        return source
    return detect_cpu_temp_from_hwmon()

def read_cpu_temp_from_source(source: Optional[dict[str, object]]) -> str:
    if source is None:
        return "N/A"

    temp_paths = source.get("temp_paths")
    if not isinstance(temp_paths, list):
        return "N/A"

    values: list[float] = []
    for temp_path in temp_paths:
        if not isinstance(temp_path, str):
            continue
        value = parse_temp_c(read_text_file(temp_path))
        if value is not None:
            values.append(value)

    if not values:
        return "N/A"
    return format_temp_c(max(values))

def read_cpu_temp_cached(source: Optional[dict[str, object]]) -> tuple[str, Optional[dict[str, object]]]:
    if source is None:
        return "N/A", None
    return read_cpu_temp_from_source(source), source

def read_cpu_freqs() -> Tuple[str, str]:
    vals: list[float] = []
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    try:
                        vals.append(float(line.split(":", 1)[1].strip()))
                    except Exception:
                        pass
    except Exception:
        return "N/A", "N/A"

    if not vals:
        return "N/A", "N/A"
    avg = f"{sum(vals) / len(vals):.0f}"
    mx = f"{max(vals):.0f}"
    return avg, mx

def read_ram() -> Tuple[str, str]:
    out = run_cmd(["free", "-m"])
    for line in out.splitlines():
        if line.strip().startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 3:
                return strip_if_na(parts[1]), strip_if_na(parts[2])
    return "N/A", "N/A"

def read_gpu_metrics() -> Tuple[str, dict[str, str]]:
    out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,power.draw,clocks.gr,clocks.mem,temperature.gpu,memory.used,memory.total,fan.speed",
            "--format=csv,noheader,nounits",
        ]
    )

    fields = ["GPU_UTIL", "GPU_POWER", "GPU_CORE", "GPU_MEMCLK", "GPU_TEMP", "VRAM_USED", "VRAM_TOTAL", "GPU_FAN"]
    na = {k: "N/A" for k in fields}

    line = out.splitlines()[0].strip() if out.splitlines() else ""
    if not line:
        return "N/A", na

    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 9:
        return "N/A", na

    model = strip_if_na(parts[0])
    nums = [p.replace(",", ".") for p in parts[1:9]]

    return model, {
        "GPU_UTIL": strip_if_na(nums[0]),
        "GPU_POWER": strip_if_na(nums[1]),
        "GPU_CORE": strip_if_na(nums[2]),
        "GPU_MEMCLK": strip_if_na(nums[3]),
        "GPU_TEMP": strip_if_na(nums[4]),
        "VRAM_USED": strip_if_na(nums[5]),
        "VRAM_TOTAL": strip_if_na(nums[6]),
        "GPU_FAN": strip_if_na(nums[7]),
    }

def format_cpu_power(prev_energy: Optional[int], curr_energy: Optional[int]) -> str:
    if prev_energy is None or curr_energy is None:
        return "N/A"
    delta = curr_energy - prev_energy
    if delta < 0:
        # RAPL counter can reset/wrap; skip such sample instead of showing negative spike.
        return "N/A"
    return f"{delta / 1_000_000:.1f}"

def empty_gpu_metrics() -> dict[str, str]:
    return {
        "GPU_UTIL": "N/A",
        "GPU_POWER": "N/A",
        "GPU_CORE": "N/A",
        "GPU_MEMCLK": "N/A",
        "GPU_TEMP": "N/A",
        "VRAM_USED": "N/A",
        "VRAM_TOTAL": "N/A",
        "GPU_FAN": "N/A",
    }

def init_metric_history() -> dict[str, deque[Optional[float]]]:
    keys = [
        "cpu_load",
        "cpu_temp",
        "cpu_power",
        "cpu_freq_avg",
        "cpu_freq_max",
        "ram_used",
        "gpu_util",
        "gpu_temp",
        "gpu_power",
        "gpu_core",
        "gpu_memclk",
        "gpu_fan",
        "vram_used",
    ]
    return {k: deque(maxlen=GRAPH_POINTS) for k in keys}

def init_metric_minmax() -> dict[str, dict[str, Optional[float]]]:
    keys = [
        "cpu_load",
        "cpu_temp",
        "cpu_power",
        "cpu_freq_avg",
        "cpu_freq_max",
        "ram_used",
        "gpu_util",
        "gpu_temp",
        "gpu_power",
        "gpu_fan",
        "gpu_core",
        "gpu_memclk",
        "vram_used",
    ]
    return {k: {"min": None, "max": None} for k in keys}

def update_metric_minmax_entry(minmax: dict[str, dict[str, Optional[float]]], key: str, raw_value: str) -> None:
    value = to_float_or_none(raw_value)
    if value is None:
        return

    slot = minmax[key]
    current_min = slot["min"]
    current_max = slot["max"]
    if current_min is None or value < current_min:
        slot["min"] = value
    if current_max is None or value > current_max:
        slot["max"] = value

def update_metric_minmax(
    minmax: dict[str, dict[str, Optional[float]]],
    cpu_load: str,
    cpu_temp: str,
    cpu_power: str,
    cpu_freq_avg: str,
    cpu_freq_max: str,
    ram_used: str,
    gpu: dict[str, str],
) -> None:
    update_metric_minmax_entry(minmax, "cpu_load", cpu_load)
    update_metric_minmax_entry(minmax, "cpu_temp", cpu_temp)
    update_metric_minmax_entry(minmax, "cpu_power", cpu_power)
    update_metric_minmax_entry(minmax, "cpu_freq_avg", cpu_freq_avg)
    update_metric_minmax_entry(minmax, "cpu_freq_max", cpu_freq_max)
    update_metric_minmax_entry(minmax, "ram_used", ram_used)
    update_metric_minmax_entry(minmax, "gpu_util", gpu["GPU_UTIL"])
    update_metric_minmax_entry(minmax, "gpu_temp", gpu["GPU_TEMP"])
    update_metric_minmax_entry(minmax, "gpu_power", gpu["GPU_POWER"])
    update_metric_minmax_entry(minmax, "gpu_fan", gpu["GPU_FAN"])
    update_metric_minmax_entry(minmax, "gpu_core", gpu["GPU_CORE"])
    update_metric_minmax_entry(minmax, "gpu_memclk", gpu["GPU_MEMCLK"])
    update_metric_minmax_entry(minmax, "vram_used", gpu["VRAM_USED"])

def push_metric_history(
    history: dict[str, deque[Optional[float]]],
    cpu_load: str,
    cpu_temp: str,
    cpu_power: str,
    cpu_freq_avg: str,
    cpu_freq_max: str,
    ram_used: str,
    gpu: dict[str, str],
) -> None:
    history["cpu_load"].append(to_float_or_none(cpu_load))
    history["cpu_temp"].append(to_float_or_none(cpu_temp))
    history["cpu_power"].append(to_float_or_none(cpu_power))
    history["cpu_freq_avg"].append(to_float_or_none(cpu_freq_avg))
    history["cpu_freq_max"].append(to_float_or_none(cpu_freq_max))
    history["ram_used"].append(to_float_or_none(ram_used))
    history["gpu_util"].append(to_float_or_none(gpu["GPU_UTIL"]))
    history["gpu_temp"].append(to_float_or_none(gpu["GPU_TEMP"]))
    history["gpu_power"].append(to_float_or_none(gpu["GPU_POWER"]))
    history["gpu_core"].append(to_float_or_none(gpu["GPU_CORE"]))
    history["gpu_memclk"].append(to_float_or_none(gpu["GPU_MEMCLK"]))
    history["gpu_fan"].append(to_float_or_none(gpu["GPU_FAN"]))
    history["vram_used"].append(to_float_or_none(gpu["VRAM_USED"]))

def max_seen(history: dict[str, deque[Optional[float]]], key: str, default: float) -> float:
    vals = [x for x in history[key] if x is not None]
    if not vals:
        return default
    return max(default, max(vals))

def sparkline_rows(values: deque[Optional[float]], width: int, scale_max: float, height: int) -> list[str]:
    data = list(values)[-width:]
    if len(data) < width:
        data = [None] * (width - len(data)) + data

    upper = max(scale_max, 1.0)
    levels = max(1, height * 8)
    steps: list[Optional[int]] = []
    for v in data:
        if v is None:
            steps.append(None)
            continue
        ratio = max(0.0, min(1.0, v / upper))
        if ratio <= 0:
            # Draw baseline even for zeros to keep lines visually continuous.
            steps.append(1)
        else:
            # Keep very low non-zero values visible to avoid broken-looking lines at 1-5%.
            steps.append(max(1, int(math.ceil(ratio * levels))))

    out_rows: list[str] = []
    for row in range(height - 1, -1, -1):
        line_chars: list[str] = []
        base = row * 8
        for st in steps:
            if st is None:
                line_chars.append(" ")
                continue
            cell = max(0, min(8, st - base))
            line_chars.append(SPARK_CHARS[cell])
        out_rows.append("".join(line_chars))
    return out_rows

