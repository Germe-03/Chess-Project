"""
Rechner wach halten, solange ein langes Training läuft.

Ein Lauf wie v5 oder v7 dauert Stunden. Windows schickt den Rechner nach einer
Weile ohne Eingabe in den Standby, und ab dann stehen die Prozesse still. So
geschehen beim ersten Versuch mit v7: Nach 19 Partien ruhte das Training von
21:32 bis zum nächsten Morgen.

SetThreadExecutionState mit ES_SYSTEM_REQUIRED meldet dem System, dass gerade
gearbeitet wird — dasselbe, was Videoplayer tun. Es ändert keine Einstellung
und endet von selbst mit dem Prozess. Gegen Zuklappen oder «Energie sparen»
von Hand hilft es bewusst nicht. Auf anderen Systemen tut es nichts.

Aufruf für einen Prozess, der schon läuft (etwa ein Training, das noch ohne
diese Funktion gestartet wurde):

    python -m Code.rl.keep_awake --pid 12345

Der Helfer hält den Rechner wach, bis dieser Prozess endet.
"""
from __future__ import annotations

import argparse
import ctypes
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
SYNCHRONIZE = 0x00100000
INFINITE = 0xFFFFFFFF


def hold() -> bool:
    """Standby verhindern, bis release() oder das Prozessende. Gibt an, ob es wirkt."""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))


def release() -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rechner wach halten, bis ein Prozess endet")
    parser.add_argument("--pid", type=int, required=True, help="Prozess, auf dessen Ende gewartet wird")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("Nur unter Windows nötig.")

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, args.pid)
    if not handle:
        raise SystemExit(f"Prozess {args.pid} nicht gefunden.")
    hold()
    print(f"Rechner bleibt wach, bis Prozess {args.pid} endet. Abbrechen mit Ctrl+C.", flush=True)
    try:
        # In kurzen Abschnitten warten, damit Ctrl+C durchkommt.
        while kernel32.WaitForSingleObject(handle, 1000) != 0:
            pass
        print(f"Prozess {args.pid} ist beendet, der Rechner darf wieder schlafen.")
    except KeyboardInterrupt:
        pass
    finally:
        release()
        kernel32.CloseHandle(handle)


if __name__ == "__main__":
    main()
