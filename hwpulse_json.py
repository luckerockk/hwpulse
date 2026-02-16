#!/usr/bin/env python3
import json
import os
import select
import signal
import sys
import time
from typing import Optional

from hwpulse_common import CPU_RAPL, format_load_pct, need_cmd, strip_if_na, to_float_or_none
from hwpulse_metrics import (
    detect_cpu_temp_source,
    empty_gpu_metrics,
    format_cpu_power,
    get_cpu_model,
    get_system_info,
    read_cpu_energy,
    read_cpu_freqs,
    read_cpu_stat,
    read_cpu_temp_cached,
    read_gpu_metrics,
    read_ram,
)
from hwpulse_terminal import cleanup_and_exit

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

def detect_output_sources() -> tuple[dict[str, object], dict[str, bool]]:
    cpu_temp_source = detect_cpu_temp_source()
    cpu_freq_avg_probe, cpu_freq_max_probe = read_cpu_freqs()
    cpu_freq_available = cpu_freq_avg_probe != "N/A" or cpu_freq_max_probe != "N/A"

    sources: dict[str, object] = {
        "cpu_stat": os.path.isfile("/proc/stat"),
        "cpu_power": os.path.isfile(CPU_RAPL),
        "cpu_temp_source": cpu_temp_source,
        "cpu_freq": cpu_freq_available,
        "ram": need_cmd("free"),
        "gpu": need_cmd("nvidia-smi"),
    }

    metric_enabled = {
        "cpu_load": bool(sources["cpu_stat"]),
        "cpu_temp": cpu_temp_source is not None,
        "cpu_power": bool(sources["cpu_power"]),
        "cpu_freq_avg": bool(sources["cpu_freq"]),
        "cpu_freq_max": bool(sources["cpu_freq"]),
        "ram_used": bool(sources["ram"]),
        "gpu_util": bool(sources["gpu"]),
        "gpu_temp": bool(sources["gpu"]),
        "gpu_power": bool(sources["gpu"]),
        "gpu_fan": bool(sources["gpu"]),
        "gpu_core": bool(sources["gpu"]),
        "gpu_memclk": bool(sources["gpu"]),
        "vram_used": bool(sources["gpu"]),
    }
    return sources, metric_enabled

def init_output_state(sources: dict[str, object]) -> dict[str, object]:
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

    cpu_temp_source = sources.get("cpu_temp_source")
    if isinstance(cpu_temp_source, dict):
        cpu_temp, cpu_temp_source = read_cpu_temp_cached(cpu_temp_source)
        sources["cpu_temp_source"] = cpu_temp_source
        state["cpu_temp"] = cpu_temp
    else:
        state["cpu_temp"] = "N/A"

    if bool(sources["cpu_freq"]):
        cpu_freq_avg, cpu_freq_max = read_cpu_freqs()
        state["cpu_freq_avg"] = cpu_freq_avg
        state["cpu_freq_max"] = cpu_freq_max
    if bool(sources["ram"]):
        ram_total, ram_used = read_ram()
        state["ram_total"] = ram_total
        state["ram_used"] = ram_used
    if bool(sources["gpu"]):
        new_gpu_model, gpu = read_gpu_metrics()
        if new_gpu_model != "N/A":
            state["gpu_model"] = new_gpu_model
        state["gpu"] = gpu

    if bool(sources["cpu_stat"]):
        prev_total, prev_idle = read_cpu_stat()
        state["prev_total"] = prev_total
        state["prev_idle"] = prev_idle
    state["prev_energy"] = read_cpu_energy() if bool(sources["cpu_power"]) else None

    return state

def sample_output_state(state: dict[str, object], sources: dict[str, object]) -> None:
    if bool(sources["cpu_stat"]):
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

    if bool(sources["cpu_power"]):
        curr_energy = read_cpu_energy()
        prev_energy = state["prev_energy"] if isinstance(state["prev_energy"], int) else None
        state["cpu_power"] = format_cpu_power(prev_energy, curr_energy)
        state["prev_energy"] = curr_energy
    else:
        state["cpu_power"] = "N/A"

    cpu_temp_source = sources.get("cpu_temp_source")
    if isinstance(cpu_temp_source, dict):
        cpu_temp, cpu_temp_source = read_cpu_temp_cached(cpu_temp_source)
        sources["cpu_temp_source"] = cpu_temp_source
        state["cpu_temp"] = cpu_temp
    else:
        state["cpu_temp"] = "N/A"

    if bool(sources["cpu_freq"]):
        cpu_freq_avg, cpu_freq_max = read_cpu_freqs()
        state["cpu_freq_avg"] = cpu_freq_avg
        state["cpu_freq_max"] = cpu_freq_max
    else:
        state["cpu_freq_avg"] = "N/A"
        state["cpu_freq_max"] = "N/A"

    if bool(sources["ram"]):
        ram_total, ram_used = read_ram()
        state["ram_total"] = ram_total
        state["ram_used"] = ram_used
    else:
        state["ram_total"] = "N/A"
        state["ram_used"] = "N/A"

    if bool(sources["gpu"]):
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

