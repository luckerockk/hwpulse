#!/usr/bin/env python3
import argparse
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import time
import math
from collections import deque
from itertools import zip_longest
from typing import Optional, Tuple

CPU_RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"

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

HAVE_TPUT = shutil.which("tput") is not None
TERM_FD: Optional[int] = None
TERM_OLD = None
SPARK_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
GRAPH_POINTS = 120
GRAPH_HEIGHT = 2
GRAPH_ROW_GAP = 1
GRAPH_UPDATE_TICKS = 2
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def need_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hwpulse",
        description="Real-time system monitor with optional terminal graphs.",
        epilog="Example:\n  hwpulse --nograph\n  hwpulse --json-stdio\n  hwpulse --json-once",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--nograph",
        action="store_true",
        help="Disable graph rendering and show only textual metrics.",
    )
    parser.add_argument(
        "--json-stdio",
        action="store_true",
        help="Run without UI; read stdin and reply with JSON on 'get'.",
    )
    parser.add_argument(
        "--json-once",
        action="store_true",
        help="Collect one snapshot, print JSON to stdout, and exit.",
    )
    args = parser.parse_args()

    if args.json_stdio and args.json_once:
        parser.error("Use only one JSON mode: --json-stdio or --json-once.")

    if args.nograph and (args.json_stdio or args.json_once):
        parser.error("--nograph is only valid for UI mode.")

    return args


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


def set_cursor(visible: bool) -> None:
    if HAVE_TPUT:
        cmd = ["tput", "cnorm" if visible else "civis"]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass
    # Force cursor state via ANSI too (works even if tput is ineffective for this terminal).
    sys.stdout.write(f"{CSI}?25{'h' if visible else 'l'}")
    sys.stdout.flush()


def set_line_wrap(enabled: bool) -> None:
    if not sys.stdout.isatty():
        return
    sys.stdout.write(f"{CSI}?7{'h' if enabled else 'l'}")
    sys.stdout.flush()


def enter_alt_screen() -> None:
    if not sys.stdout.isatty():
        return
    sys.stdout.write(f"{CSI}?1049h{CSI}H")
    sys.stdout.flush()


def leave_alt_screen() -> None:
    if not sys.stdout.isatty():
        return
    sys.stdout.write(f"{CSI}?1049l")
    sys.stdout.flush()


def cleanup_and_exit(_sig: int, _frame) -> None:
    raise SystemExit(0)


def enable_input_mode() -> None:
    global TERM_FD, TERM_OLD
    if not sys.stdin.isatty():
        return
    TERM_FD = sys.stdin.fileno()
    TERM_OLD = termios.tcgetattr(TERM_FD)
    attrs = termios.tcgetattr(TERM_FD)
    attrs[3] &= ~(termios.ICANON | termios.ECHO)
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(TERM_FD, termios.TCSANOW, attrs)


def restore_input_mode() -> None:
    global TERM_FD, TERM_OLD
    if TERM_FD is None or TERM_OLD is None:
        return
    try:
        termios.tcsetattr(TERM_FD, termios.TCSANOW, TERM_OLD)
    except Exception:
        pass
    TERM_FD = None
    TERM_OLD = None


def should_exit_now() -> bool:
    if TERM_FD is None:
        return False
    try:
        ready, _, _ = select.select([TERM_FD], [], [], 0)
    except Exception:
        return False
    if not ready:
        return False
    try:
        data = os.read(TERM_FD, 64)
    except Exception:
        return False
    if not data:
        return False
    if data == b"\x1b":
        # Some terminal control sequences may arrive split. Give them a tiny
        # chance to complete before treating Esc as an exit key.
        try:
            ready2, _, _ = select.select([TERM_FD], [], [], 0.01)
            if ready2:
                data += os.read(TERM_FD, 64)
        except Exception:
            pass
    return has_standalone_esc(data)


def has_standalone_esc(data: bytes) -> bool:
    i = 0
    n = len(data)
    while i < n:
        if data[i] != 0x1B:
            i += 1
            continue

        # Plain Esc key (single byte) => exit.
        if i == n - 1:
            return True

        nxt = data[i + 1]
        # CSI / SS3 sequences (arrows, function keys, mouse, etc.) should not exit.
        if nxt in (ord("["), ord("O")):
            i += 2
            while i < n:
                b = data[i]
                if 0x40 <= b <= 0x7E:
                    i += 1
                    break
                i += 1
            continue

        # Alt-modified key sequence (Esc + key) should not exit.
        i += 2

    return False


def wait_or_exit(seconds: float) -> bool:
    end_at = time.monotonic() + seconds
    while time.monotonic() < end_at:
        if should_exit_now():
            return True
        time.sleep(min(0.05, end_at - time.monotonic()))
    return False


def parse_cpu_temp(sensors_out: str) -> str:
    for line in sensors_out.splitlines():
        if "Tctl" in line:
            m = re.search(r"Tctl:\s*\+?([0-9]+(?:[\.,][0-9]+)?)", line)
            if m:
                return m.group(1).replace(",", ".")
    return "N/A"


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


def to_float_or_none(value: str) -> Optional[float]:
    raw = strip_if_na(value)
    if raw == "N/A":
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


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


def format_latest(raw: str, unit: str) -> str:
    value = strip_if_na(raw)
    if value == "N/A":
        return "N/A"
    return f"{value} {unit}"


def format_with_unit(raw: str, rendered: str, unit: str) -> str:
    if strip_if_na(raw) == "N/A":
        return "N/A"
    return f"{rendered} {unit}"


def format_label_with_total(label: str, total_raw: str) -> str:
    total = strip_if_na(total_raw)
    if total == "N/A":
        return label
    return f"{label} ({total})"


def format_load_pct(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.0f}"


def format_metric_stat_value(metric_key: str, value: Optional[float]) -> str:
    if value is None:
        return "N/A"

    if metric_key in ("cpu_load", "gpu_util", "gpu_fan"):
        return format_load_pct(value)

    if metric_key in ("cpu_temp", "gpu_temp"):
        return f"{value:.1f}".rstrip("0").rstrip(".")

    if metric_key in ("cpu_power", "gpu_power"):
        return f"{value:.2f}".rstrip("0").rstrip(".")

    if metric_key in ("cpu_freq_avg", "cpu_freq_max", "gpu_core", "gpu_memclk", "ram_used", "vram_used"):
        return f"{value:.0f}"

    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_stat_with_unit(metric_key: str, value: Optional[float], unit: str) -> str:
    base = format_metric_stat_value(metric_key, value)
    if base == "N/A":
        return "N/A"
    return f"{base} {unit}"


def visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def rjust_visible(text: str, width: int) -> str:
    pad = max(0, width - visible_len(text))
    return (" " * pad) + text


def ljust_visible(text: str, width: int) -> str:
    pad = max(0, width - visible_len(text))
    return text + (" " * pad)


def graph_entry_lines(
    label: str,
    history: deque[Optional[float]],
    scale_max: float,
    tail_value: str,
    graph_width: int,
    graph_height: int,
) -> list[str]:
    spark_rows = sparkline_rows(history, graph_width, scale_max, graph_height)
    out: list[str] = []
    for i, spark in enumerate(spark_rows):
        prefix = f"{label:<10} " if i == 0 else " " * 11
        suffix = f" {rjust_visible(tail_value, 9)}" if i == len(spark_rows) - 1 else " " * 10
        out.append(f"{prefix}{spark}{suffix}")
    return out


def render_line(text: str) -> str:
    return f"{text}{CSI}K\n"


def render_metric_table_lines(
    section: str,
    rows: list[tuple[str, str, str, str]],
    metric_minmax: dict[str, dict[str, Optional[float]]],
) -> list[str]:
    label_w = max(11, max((visible_len(label) for label, _, _, _ in rows), default=0))
    current_w = 24
    min_w = 12
    max_w = 12

    out: list[str] = []
    out.append(render_line(f"{STYLE_SECTION}{section}{CLR_RESET}"))
    out.append(
        render_line(
            f"{STYLE_SECTION}  {'Metric':<{label_w}} {'Current':>{current_w}} {'Min':>{min_w}} {'Max':>{max_w}}{CLR_RESET}"
        )
    )

    for label, current_text, metric_key, unit in rows:
        slot = metric_minmax[metric_key]
        min_text = format_stat_with_unit(metric_key, slot["min"], unit)
        max_text = format_stat_with_unit(metric_key, slot["max"], unit)
        line = (
            f"  {ljust_visible(label, label_w)} "
            f"{rjust_visible(current_text, current_w)} "
            f"{rjust_visible(min_text, min_w)} "
            f"{rjust_visible(max_text, max_w)}"
        )
        out.append(render_line(line))

    out.append(render_line(""))
    return out


def render_dashboard(
    system_name: str,
    cpu_model: str,
    gpu_model: str,
    cpu_load: str,
    cpu_temp: str,
    cpu_power: str,
    cpu_freq_avg: str,
    cpu_freq_max: str,
    ram_used: str,
    ram_total: str,
    gpu: dict[str, str],
    metric_minmax: dict[str, dict[str, Optional[float]]],
    history: Optional[dict[str, deque[Optional[float]]]],
    update_graphs: bool,
    show_graphs: bool,
    metric_enabled: dict[str, bool],
) -> None:
    update_metric_minmax(metric_minmax, cpu_load, cpu_temp, cpu_power, cpu_freq_avg, cpu_freq_max, ram_used, gpu)

    if show_graphs and update_graphs and history is not None:
        push_metric_history(history, cpu_load, cpu_temp, cpu_power, cpu_freq_avg, cpu_freq_max, ram_used, gpu)

    cpu_load_c = color_pct(strip_if_na(cpu_load))
    gpu_util_c = color_pct(strip_if_na(gpu["GPU_UTIL"]))

    cpu_temp_c = color_temp(strip_if_na(cpu_temp))
    gpu_temp_c = color_temp(strip_if_na(gpu["GPU_TEMP"]))

    cpu_power_c = color_power_cpu(strip_if_na(cpu_power))
    gpu_power_c = color_power_gpu(strip_if_na(gpu["GPU_POWER"]))

    cpu_load_text = format_with_unit(cpu_load, cpu_load_c, "%")
    cpu_temp_text = format_with_unit(cpu_temp, cpu_temp_c, f"\N{DEGREE SIGN}C")
    cpu_power_text = format_with_unit(cpu_power, cpu_power_c, "W")
    cpu_freq_avg_text = format_latest(cpu_freq_avg, "MHz")
    cpu_freq_max_text = format_latest(cpu_freq_max, "MHz")
    ram_used_text = format_latest(ram_used, "MB")
    ram_label = format_label_with_total("RAM", ram_total)

    gpu_load_text = format_with_unit(gpu["GPU_UTIL"], gpu_util_c, "%")
    gpu_temp_text = format_with_unit(gpu["GPU_TEMP"], gpu_temp_c, f"\N{DEGREE SIGN}C")
    gpu_power_text = format_with_unit(gpu["GPU_POWER"], gpu_power_c, "W")
    gpu_core_text = format_latest(gpu["GPU_CORE"], "MHz")
    gpu_mem_text = format_latest(gpu["GPU_MEMCLK"], "MHz")
    gpu_fan_text = format_latest(gpu["GPU_FAN"], "%")
    vram_used_text = format_latest(gpu["VRAM_USED"], "MB")
    vram_label = format_label_with_total("VRAM", gpu["VRAM_TOTAL"])

    cpu_load_tail = cpu_load_text
    cpu_temp_tail = cpu_temp_text
    cpu_power_tail = cpu_power_text
    cpu_freq_avg_tail = cpu_freq_avg_text
    cpu_freq_max_tail = cpu_freq_max_text
    ram_used_tail = ram_used_text

    gpu_load_tail = gpu_load_text
    gpu_temp_tail = gpu_temp_text
    gpu_power_tail = gpu_power_text
    gpu_core_tail = gpu_core_text
    gpu_mem_tail = gpu_mem_text
    gpu_fan_tail = gpu_fan_text
    vram_used_tail = vram_used_text

    term_size = shutil.get_terminal_size((120, 40))
    term_cols = term_size.columns
    term_rows = max(1, term_size.lines)
    max_header_width = max(20, term_cols - 1)

    header_title = "HWPULSE"
    header_system = f"SYSTEM: {system_name}"
    header_cpu = f"CPU: {cpu_model}"
    header_gpu = f"GPU: {gpu_model}"

    header_title = header_title[:max_header_width]
    header_system = header_system[:max_header_width]
    header_cpu = header_cpu[:max_header_width]
    header_gpu = header_gpu[:max_header_width]
    line = "-" * max(len(header_title), len(header_system), len(header_cpu), len(header_gpu))

    body_lines: list[str] = []
    body_lines.append(render_line(f"{STYLE_TITLE}{header_title}{CLR_RESET}"))
    body_lines.append(render_line(f"{STYLE_TITLE}{line}{CLR_RESET}"))
    body_lines.append(render_line(f"{STYLE_TITLE}{header_system}{CLR_RESET}"))
    body_lines.append(render_line(f"{STYLE_TITLE}{header_cpu}{CLR_RESET}"))
    body_lines.append(render_line(f"{STYLE_TITLE}{header_gpu}{CLR_RESET}"))
    body_lines.append(render_line(f"{STYLE_TITLE}{line}{CLR_RESET}"))
    cpu_rows = [
        ("Load", cpu_load_text, "cpu_load", "%"),
        ("Temp", cpu_temp_text, "cpu_temp", f"{chr(176)}C"),
        ("Power", cpu_power_text, "cpu_power", "W"),
        ("Freq (Avg)", cpu_freq_avg_text, "cpu_freq_avg", "MHz"),
        ("Freq (Max)", cpu_freq_max_text, "cpu_freq_max", "MHz"),
        (ram_label, ram_used_text, "ram_used", "MB"),
    ]
    gpu_rows = [
        ("Load", gpu_load_text, "gpu_util", "%"),
        ("Temp", gpu_temp_text, "gpu_temp", f"{chr(176)}C"),
        ("Power", gpu_power_text, "gpu_power", "W"),
        ("GPU FAN", gpu_fan_text, "gpu_fan", "%"),
        ("Core", gpu_core_text, "gpu_core", "MHz"),
        ("Mem Freq", gpu_mem_text, "gpu_memclk", "MHz"),
        (vram_label, vram_used_text, "vram_used", "MB"),
    ]

    body_lines.extend(render_metric_table_lines("CPU", cpu_rows, metric_minmax))
    body_lines.extend(render_metric_table_lines("GPU", gpu_rows, metric_minmax))

    if show_graphs and history is not None:
        cols = term_cols
        sep = "  |  "
        col_width = max(34, (cols - len(sep)) // 2)
        graph_width = max(10, col_width - 21)

        ram_scale = to_float_or_none(ram_total) or max_seen(history, "ram_used", 1.0)
        vram_scale = to_float_or_none(gpu["VRAM_TOTAL"]) or max_seen(history, "vram_used", 1.0)
        cpu_freq_scale = max_seen(history, "cpu_freq_max", 6000.0)
        gpu_core_scale = max_seen(history, "gpu_core", 3500.0)
        gpu_mem_scale = max_seen(history, "gpu_memclk", 15000.0)

        left_blocks: list[list[str]] = []
        if metric_enabled["cpu_load"]:
            left_blocks.append(graph_entry_lines("CPU Load", history["cpu_load"], 100.0, cpu_load_tail, graph_width, GRAPH_HEIGHT))
        if metric_enabled["cpu_temp"]:
            left_blocks.append(graph_entry_lines("CPU Temp", history["cpu_temp"], 100.0, cpu_temp_tail, graph_width, GRAPH_HEIGHT))
        if metric_enabled["cpu_power"]:
            left_blocks.append(graph_entry_lines("CPU Power", history["cpu_power"], 220.0, cpu_power_tail, graph_width, GRAPH_HEIGHT))
        if metric_enabled["cpu_freq_avg"]:
            left_blocks.append(
                graph_entry_lines(
                    "CPU FreqA",
                    history["cpu_freq_avg"],
                    cpu_freq_scale,
                    cpu_freq_avg_tail,
                    graph_width,
                    GRAPH_HEIGHT,
                )
            )
        if metric_enabled["cpu_freq_max"]:
            left_blocks.append(
                graph_entry_lines(
                    "CPU FreqM",
                    history["cpu_freq_max"],
                    cpu_freq_scale,
                    cpu_freq_max_tail,
                    graph_width,
                    GRAPH_HEIGHT,
                )
            )
        if metric_enabled["ram_used"]:
            left_blocks.append(graph_entry_lines("RAM Used", history["ram_used"], ram_scale, ram_used_tail, graph_width, GRAPH_HEIGHT))

        right_blocks: list[list[str]] = []
        if metric_enabled["gpu_util"]:
            right_blocks.append(graph_entry_lines("GPU Load", history["gpu_util"], 100.0, gpu_load_tail, graph_width, GRAPH_HEIGHT))
        if metric_enabled["gpu_temp"]:
            right_blocks.append(graph_entry_lines("GPU Temp", history["gpu_temp"], 100.0, gpu_temp_tail, graph_width, GRAPH_HEIGHT))
        if metric_enabled["gpu_power"]:
            right_blocks.append(graph_entry_lines("GPU Power", history["gpu_power"], 650.0, gpu_power_tail, graph_width, GRAPH_HEIGHT))
        if metric_enabled["gpu_core"]:
            right_blocks.append(
                graph_entry_lines(
                    "GPU Core",
                    history["gpu_core"],
                    gpu_core_scale,
                    gpu_core_tail,
                    graph_width,
                    GRAPH_HEIGHT,
                )
            )
        if metric_enabled["gpu_fan"]:
            right_blocks.append(
                graph_entry_lines(
                    "GPU Fan",
                    history["gpu_fan"],
                    100.0,
                    gpu_fan_tail,
                    graph_width,
                    GRAPH_HEIGHT,
                )
            )
        if metric_enabled["gpu_memclk"]:
            right_blocks.append(
                graph_entry_lines(
                    "GPU MemFrq",
                    history["gpu_memclk"],
                    gpu_mem_scale,
                    gpu_mem_tail,
                    graph_width,
                    GRAPH_HEIGHT,
                )
            )
        if metric_enabled["vram_used"]:
            right_blocks.append(graph_entry_lines("VRAM Used", history["vram_used"], vram_scale, vram_used_tail, graph_width, GRAPH_HEIGHT))

        if left_blocks or right_blocks:
            body_lines.append(render_line(f"{STYLE_SECTION}GRAPHS{CLR_RESET}"))
            body_lines.append(render_line(""))

            blank_line = " " * (graph_width + 21)
            blank_block = [blank_line] * GRAPH_HEIGHT

            for left_block, right_block in zip_longest(left_blocks, right_blocks, fillvalue=blank_block):
                for left_line, right_line in zip(left_block, right_block):
                    row = f"{left_line}{sep}{right_line}"
                    body_lines.append(render_line(row))
                for _ in range(GRAPH_ROW_GAP):
                    body_lines.append(render_line(""))

            body_lines.append(render_line(""))

    body = body_lines[:term_rows]
    if body:
        body[-1] = body[-1].rstrip("\n")

    frame = ["\033[?25l\033[H"]
    frame.extend(body)
    frame.append(f"{CSI}J")
    sys.stdout.write("".join(frame))
    sys.stdout.flush()


def build_output_payload(
    system_name: str,
    kernel_version: str,
    cpu_model: str,
    gpu_model: str,
    cpu_load: str,
    cpu_temp: str,
    cpu_power: str,
    cpu_freq_avg: str,
    cpu_freq_max: str,
    ram_used: str,
    ram_total: str,
    gpu: dict[str, str],
    metric_enabled: dict[str, bool],
) -> dict:
    cpu_obj: dict[str, object] = {"modelName": cpu_model}
    if metric_enabled["cpu_load"]:
        cpu_obj["loadPercent"] = to_float_or_none(cpu_load)
    if metric_enabled["cpu_temp"]:
        cpu_obj["tempC"] = to_float_or_none(cpu_temp)
    if metric_enabled["cpu_power"]:
        cpu_obj["powerW"] = to_float_or_none(cpu_power)
    if metric_enabled["cpu_freq_avg"]:
        cpu_obj["freqAvgMhz"] = to_float_or_none(cpu_freq_avg)
    if metric_enabled["cpu_freq_max"]:
        cpu_obj["freqMaxMhz"] = to_float_or_none(cpu_freq_max)
    if metric_enabled["ram_used"]:
        cpu_obj["ramUsedMb"] = to_float_or_none(ram_used)
        cpu_obj["ramTotalMb"] = to_float_or_none(ram_total)

    payload: dict[str, object] = {
        "timestamp": int(time.time() * 1000),
        "systemName": system_name,
        "kernel": kernel_version,
        "cpu": cpu_obj,
    }

    gpu_model_name = strip_if_na(gpu_model)
    gpu_supported = any(
        metric_enabled[k]
        for k in ("gpu_util", "gpu_temp", "gpu_power", "gpu_fan", "gpu_core", "gpu_memclk", "vram_used")
    )
    if gpu_supported and gpu_model_name != "N/A":
        gpu_obj: dict[str, object] = {"modelName": gpu_model_name}
        if metric_enabled["gpu_util"]:
            gpu_obj["loadPercent"] = to_float_or_none(gpu["GPU_UTIL"])
        if metric_enabled["gpu_temp"]:
            gpu_obj["tempC"] = to_float_or_none(gpu["GPU_TEMP"])
        if metric_enabled["gpu_power"]:
            gpu_obj["powerW"] = to_float_or_none(gpu["GPU_POWER"])
        if metric_enabled["gpu_fan"]:
            gpu_obj["fanPercent"] = to_float_or_none(gpu["GPU_FAN"])
        if metric_enabled["gpu_core"]:
            gpu_obj["coreMhz"] = to_float_or_none(gpu["GPU_CORE"])
        if metric_enabled["gpu_memclk"]:
            gpu_obj["memMhz"] = to_float_or_none(gpu["GPU_MEMCLK"])
        if metric_enabled["vram_used"]:
            gpu_obj["vramUsedMb"] = to_float_or_none(gpu["VRAM_USED"])
            gpu_obj["vramTotalMb"] = to_float_or_none(gpu["VRAM_TOTAL"])
        payload["gpu"] = gpu_obj

    return payload


def detect_output_sources() -> tuple[dict[str, bool], dict[str, bool]]:
    sources = {
        "cpu_stat": os.path.isfile("/proc/stat"),
        "cpu_power": os.path.isfile(CPU_RAPL),
        "cpu_temp": need_cmd("sensors"),
        "cpu_freq": os.path.isfile("/proc/cpuinfo"),
        "ram": need_cmd("free"),
        "gpu": need_cmd("nvidia-smi"),
    }

    metric_enabled = {
        "cpu_load": sources["cpu_stat"],
        "cpu_temp": sources["cpu_temp"],
        "cpu_power": sources["cpu_power"],
        "cpu_freq_avg": sources["cpu_freq"],
        "cpu_freq_max": sources["cpu_freq"],
        "ram_used": sources["ram"],
        "gpu_util": sources["gpu"],
        "gpu_temp": sources["gpu"],
        "gpu_power": sources["gpu"],
        "gpu_fan": sources["gpu"],
        "gpu_core": sources["gpu"],
        "gpu_memclk": sources["gpu"],
        "vram_used": sources["gpu"],
    }
    return sources, metric_enabled


def init_output_state(sources: dict[str, bool]) -> dict[str, object]:
    system_name, _, kernel_version = get_system_info()
    state: dict[str, object] = {
        "system_name": system_name,
        "kernel_version": kernel_version,
        "cpu_model": get_cpu_model(),
        "gpu_model": "N/A",
        "gpu": empty_gpu_metrics(),
        "cpu_load": "N/A",
        "cpu_power": "N/A",
        "cpu_temp": "N/A",
        "cpu_freq_avg": "N/A",
        "cpu_freq_max": "N/A",
        "ram_total": "N/A",
        "ram_used": "N/A",
        "prev_total": 0,
        "prev_idle": 0,
        "prev_energy": None,
    }

    if sources["cpu_temp"]:
        state["cpu_temp"] = parse_cpu_temp(run_cmd(["sensors"]))
    if sources["cpu_freq"]:
        cpu_freq_avg, cpu_freq_max = read_cpu_freqs()
        state["cpu_freq_avg"] = cpu_freq_avg
        state["cpu_freq_max"] = cpu_freq_max
    if sources["ram"]:
        ram_total, ram_used = read_ram()
        state["ram_total"] = ram_total
        state["ram_used"] = ram_used
    if sources["gpu"]:
        new_gpu_model, gpu = read_gpu_metrics()
        if new_gpu_model != "N/A":
            state["gpu_model"] = new_gpu_model
        state["gpu"] = gpu

    if sources["cpu_stat"]:
        prev_total, prev_idle = read_cpu_stat()
        state["prev_total"] = prev_total
        state["prev_idle"] = prev_idle
    state["prev_energy"] = read_cpu_energy() if sources["cpu_power"] else None

    return state


def sample_output_state(state: dict[str, object], sources: dict[str, bool]) -> None:
    if sources["cpu_stat"]:
        curr_total, curr_idle = read_cpu_stat()
        prev_total = int(state["prev_total"])
        prev_idle = int(state["prev_idle"])
        dt = curr_total - prev_total
        di = curr_idle - prev_idle
        if dt > 0:
            cpu_load_val = 100.0 * (dt - di) / dt
            state["cpu_load"] = format_load_pct(cpu_load_val)
        else:
            state["cpu_load"] = "N/A"
        state["prev_total"] = curr_total
        state["prev_idle"] = curr_idle
    else:
        state["cpu_load"] = "N/A"

    if sources["cpu_power"]:
        curr_energy = read_cpu_energy()
        prev_energy = state["prev_energy"] if isinstance(state["prev_energy"], int) else None
        state["cpu_power"] = format_cpu_power(prev_energy, curr_energy)
        state["prev_energy"] = curr_energy
    else:
        state["cpu_power"] = "N/A"

    if sources["cpu_temp"]:
        state["cpu_temp"] = parse_cpu_temp(run_cmd(["sensors"]))
    else:
        state["cpu_temp"] = "N/A"

    if sources["cpu_freq"]:
        cpu_freq_avg, cpu_freq_max = read_cpu_freqs()
        state["cpu_freq_avg"] = cpu_freq_avg
        state["cpu_freq_max"] = cpu_freq_max
    else:
        state["cpu_freq_avg"] = "N/A"
        state["cpu_freq_max"] = "N/A"

    if sources["ram"]:
        ram_total, ram_used = read_ram()
        state["ram_total"] = ram_total
        state["ram_used"] = ram_used
    else:
        state["ram_total"] = "N/A"
        state["ram_used"] = "N/A"

    if sources["gpu"]:
        new_gpu_model, gpu = read_gpu_metrics()
        if new_gpu_model != "N/A":
            state["gpu_model"] = new_gpu_model
        state["gpu"] = gpu
    else:
        state["gpu_model"] = "N/A"
        state["gpu"] = empty_gpu_metrics()


def build_output_payload_from_state(state: dict[str, object], metric_enabled: dict[str, bool]) -> dict:
    gpu_metrics = state["gpu"] if isinstance(state["gpu"], dict) else empty_gpu_metrics()
    return build_output_payload(
        str(state["system_name"]),
        str(state["kernel_version"]),
        str(state["cpu_model"]),
        str(state["gpu_model"]),
        str(state["cpu_load"]),
        str(state["cpu_temp"]),
        str(state["cpu_power"]),
        str(state["cpu_freq_avg"]),
        str(state["cpu_freq_max"]),
        str(state["ram_used"]),
        str(state["ram_total"]),
        gpu_metrics,
        metric_enabled,
    )


def run_json_stdio_mode() -> int:
    sources, metric_enabled = detect_output_sources()

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    state = init_output_state(sources)

    sample_interval = 1.0
    next_sample = time.monotonic() + sample_interval

    while True:
        now = time.monotonic()
        if now >= next_sample:
            sample_output_state(state, sources)
            while now >= next_sample:
                next_sample += sample_interval

        timeout = max(0.0, next_sample - time.monotonic())
        line: Optional[str] = None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                line = sys.stdin.readline()
        except Exception:
            # Fallback for environments where select() does not support stdin.
            line = sys.stdin.readline()

        if line is None:
            continue

        if line == "":
            break

        cmd = line.strip().lower()
        if cmd != "get":
            continue

        payload = build_output_payload_from_state(state, metric_enabled)
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    return 0


def run_json_once_mode() -> int:
    sources, metric_enabled = detect_output_sources()
    state = init_output_state(sources)
    time.sleep(0.2)
    sample_output_state(state, sources)
    payload = build_output_payload_from_state(state, metric_enabled)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()
    return 0


def main() -> int:
    args = parse_args()
    if args.json_stdio:
        return run_json_stdio_mode()
    if args.json_once:
        return run_json_once_mode()

    show_graphs = not args.nograph

    source_cpu_stat = os.path.isfile("/proc/stat")
    source_cpu_power = os.path.isfile(CPU_RAPL)
    source_cpu_temp = need_cmd("sensors")
    source_cpu_freq = os.path.isfile("/proc/cpuinfo")
    source_ram = need_cmd("free")
    source_gpu = need_cmd("nvidia-smi")

    metric_enabled = {
        "cpu_load": source_cpu_stat,
        "cpu_temp": source_cpu_temp,
        "cpu_power": source_cpu_power,
        "cpu_freq_avg": source_cpu_freq,
        "cpu_freq_max": source_cpu_freq,
        "ram_used": source_ram,
        "gpu_util": source_gpu,
        "gpu_temp": source_gpu,
        "gpu_power": source_gpu,
        "gpu_fan": source_gpu,
        "gpu_core": source_gpu,
        "gpu_memclk": source_gpu,
        "vram_used": source_gpu,
    }

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    system_name = get_system_name()
    cpu_model = get_cpu_model()
    gpu_model = "N/A"
    gpu = {
        "GPU_UTIL": "N/A",
        "GPU_POWER": "N/A",
        "GPU_CORE": "N/A",
        "GPU_MEMCLK": "N/A",
        "GPU_TEMP": "N/A",
        "VRAM_USED": "N/A",
        "VRAM_TOTAL": "N/A",
        "GPU_FAN": "N/A",
    }

    cpu_load = "N/A"
    cpu_power = "N/A"
    cpu_temp = "N/A"
    cpu_freq_avg = "N/A"
    cpu_freq_max = "N/A"
    ram_total = "N/A"
    ram_used = "N/A"
    metric_history = init_metric_history() if show_graphs else None
    metric_minmax = init_metric_minmax()
    graph_tick = 0

    enter_alt_screen()
    set_line_wrap(False)
    set_cursor(False)
    enable_input_mode()
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    try:
        if source_cpu_temp:
            cpu_temp = parse_cpu_temp(run_cmd(["sensors"]))
        if source_cpu_freq:
            cpu_freq_avg, cpu_freq_max = read_cpu_freqs()
        if source_ram:
            ram_total, ram_used = read_ram()
        if source_gpu:
            new_gpu_model, gpu = read_gpu_metrics()
            if new_gpu_model != "N/A":
                gpu_model = new_gpu_model
        render_dashboard(
            system_name,
            cpu_model,
            gpu_model,
            cpu_load,
            cpu_temp,
            cpu_power,
            cpu_freq_avg,
            cpu_freq_max,
            ram_used,
            ram_total,
            gpu,
            metric_minmax,
            metric_history,
            show_graphs,
            show_graphs,
            metric_enabled,
        )

        if source_cpu_stat:
            prev_total, prev_idle = read_cpu_stat()
        else:
            prev_total, prev_idle = 0, 0
        prev_energy = read_cpu_energy() if source_cpu_power else None

        while True:
            if wait_or_exit(1):
                break

            if source_cpu_stat:
                curr_total, curr_idle = read_cpu_stat()
                dt = curr_total - prev_total
                di = curr_idle - prev_idle
                if dt > 0:
                    cpu_load_val = 100.0 * (dt - di) / dt
                    cpu_load = format_load_pct(cpu_load_val)
                else:
                    cpu_load = "N/A"
                prev_total, prev_idle = curr_total, curr_idle
            else:
                cpu_load = "N/A"

            if source_cpu_power:
                curr_energy = read_cpu_energy()
                cpu_power = format_cpu_power(prev_energy, curr_energy)
                prev_energy = curr_energy
            else:
                cpu_power = "N/A"

            cpu_temp = "N/A"
            if source_cpu_temp:
                cpu_temp = parse_cpu_temp(run_cmd(["sensors"]))

            if source_cpu_freq:
                cpu_freq_avg, cpu_freq_max = read_cpu_freqs()
            else:
                cpu_freq_avg, cpu_freq_max = "N/A", "N/A"

            if source_ram:
                ram_total, ram_used = read_ram()
            else:
                ram_total, ram_used = "N/A", "N/A"

            if source_gpu:
                new_gpu_model, gpu = read_gpu_metrics()
                if new_gpu_model != "N/A":
                    gpu_model = new_gpu_model
            else:
                gpu_model = "N/A"
                gpu = {
                    "GPU_UTIL": "N/A",
                    "GPU_POWER": "N/A",
                    "GPU_CORE": "N/A",
                    "GPU_MEMCLK": "N/A",
                    "GPU_TEMP": "N/A",
                    "VRAM_USED": "N/A",
                    "VRAM_TOTAL": "N/A",
                    "GPU_FAN": "N/A",
                }

            if show_graphs:
                graph_tick += 1
                update_graphs = (graph_tick % GRAPH_UPDATE_TICKS) == 0
            else:
                update_graphs = False

            render_dashboard(
                system_name,
                cpu_model,
                gpu_model,
                cpu_load,
                cpu_temp,
                cpu_power,
                cpu_freq_avg,
                cpu_freq_max,
                ram_used,
                ram_total,
                gpu,
                metric_minmax,
                metric_history,
                update_graphs,
                show_graphs,
                metric_enabled,
            )
    finally:
        restore_input_mode()
        set_cursor(True)
        set_line_wrap(True)
        leave_alt_screen()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

