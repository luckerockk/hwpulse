#!/usr/bin/env python3
import shutil
import sys
from collections import deque
from itertools import zip_longest
from typing import Optional

from hwpulse_common import (
    ANSI_RE,
    CSI,
    CLR_RESET,
    GRAPH_HEIGHT,
    GRAPH_ROW_GAP,
    STYLE_SECTION,
    STYLE_TITLE,
    color_pct,
    color_power_cpu,
    color_power_gpu,
    color_temp,
    format_load_pct,
    strip_if_na,
    to_float_or_none,
)
from hwpulse_metrics import max_seen, push_metric_history, sparkline_rows, update_metric_minmax

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

