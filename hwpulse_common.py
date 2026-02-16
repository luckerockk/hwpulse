#!/usr/bin/env python3
import re
import shutil
import subprocess
from typing import Optional

CPU_RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
THERMAL_BASE = "/sys/class/thermal"
HWMON_BASE = "/sys/class/hwmon"
CPU_THERMAL_TYPES = {"x86_pkg_temp", "soc_thermal", "soc-thermal", "cpu_thermal", "cpu-thermal"}
CPU_HWMON_CHIPS = {"k10temp", "coretemp"}

CSI = "\033["
CLR_RESET = f"{CSI}0m"
CLR_BOLD = f"{CSI}1m"
CLR_DIM = f"{CSI}2m"
CLR_BLU = f"{CSI}34m"
CLR_RED = f"{CSI}31m"
CLR_YEL = f"{CSI}33m"
CLR_GRN = f"{CSI}32m"
CLR_CYN = f"{CSI}36m"
STYLE_TITLE = f"{CSI}1;34m"
STYLE_SECTION = f"{CSI}96m"

SPARK_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
GRAPH_POINTS = 120
GRAPH_HEIGHT = 2
GRAPH_ROW_GAP = 1
GRAPH_UPDATE_TICKS = 2
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def need_cmd(name: str) -> bool:
    return shutil.which(name) is not None

def run_cmd(args: list[str]) -> str:
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return ""
    return out

def strip_if_na(value: Optional[str]) -> str:
    if value is None:
        return "N/A"
    value = value.strip()
    return value if value else "N/A"

def color_by_thresholds(value: str, low: float, high: float) -> str:
    if value == "N/A":
        return value
    try:
        x = float(value)
    except ValueError:
        return value
    if x < low:
        return f"{CLR_GRN}{value}{CLR_RESET}"
    if x < high:
        return f"{CLR_YEL}{value}{CLR_RESET}"
    return f"{CLR_RED}{value}{CLR_RESET}"

def color_temp(value: str) -> str:
    return color_by_thresholds(value, 70.0, 85.0)

def color_pct(value: str) -> str:
    return color_by_thresholds(value, 60.0, 90.0)

def color_power_gpu(value: str) -> str:
    return color_by_thresholds(value, 300.0, 500.0)

def color_power_cpu(value: str) -> str:
    return color_by_thresholds(value, 120.0, 180.0)

def to_float_or_none(value: str) -> Optional[float]:
    raw = strip_if_na(value)
    if raw == "N/A":
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None

def format_load_pct(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.0f}"

