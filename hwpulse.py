#!/usr/bin/env python3
import argparse
import os
import signal
import sys

from hwpulse_common import CPU_RAPL, GRAPH_UPDATE_TICKS, format_load_pct, need_cmd
from hwpulse_json import run_json_once_mode, run_json_stdio_mode
from hwpulse_metrics import (
    detect_cpu_temp_source,
    format_cpu_power,
    get_cpu_model,
    get_system_name,
    init_metric_history,
    init_metric_minmax,
    read_cpu_energy,
    read_cpu_freqs,
    read_cpu_stat,
    read_cpu_temp_cached,
    read_gpu_metrics,
    read_ram,
)
from hwpulse_terminal import (
    cleanup_and_exit,
    enable_input_mode,
    enter_alt_screen,
    leave_alt_screen,
    restore_input_mode,
    set_cursor,
    set_line_wrap,
    wait_or_exit,
)
from hwpulse_ui import render_dashboard

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

def main() -> int:
    args = parse_args()
    if args.json_stdio:
        return run_json_stdio_mode()
    if args.json_once:
        return run_json_once_mode()

    show_graphs = not args.nograph

    source_cpu_stat = os.path.isfile("/proc/stat")
    source_cpu_power = os.path.isfile(CPU_RAPL)
    cpu_temp_source = detect_cpu_temp_source()
    source_cpu_temp = cpu_temp_source is not None
    cpu_freq_avg_probe, cpu_freq_max_probe = read_cpu_freqs()
    source_cpu_freq = cpu_freq_avg_probe != "N/A" or cpu_freq_max_probe != "N/A"
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
    cpu_freq_avg = cpu_freq_avg_probe if source_cpu_freq else "N/A"
    cpu_freq_max = cpu_freq_max_probe if source_cpu_freq else "N/A"
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
            cpu_temp, cpu_temp_source = read_cpu_temp_cached(cpu_temp_source)
            source_cpu_temp = cpu_temp_source is not None
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
                cpu_temp, cpu_temp_source = read_cpu_temp_cached(cpu_temp_source)
                source_cpu_temp = cpu_temp_source is not None

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
