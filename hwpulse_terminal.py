#!/usr/bin/env python3
import os
import select
import shutil
import subprocess
import sys
import termios
import time
from typing import Optional

from hwpulse_common import CSI

HAVE_TPUT = shutil.which("tput") is not None
TERM_FD: Optional[int] = None
TERM_OLD = None

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

