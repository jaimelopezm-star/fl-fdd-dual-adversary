"""
download_cwru.py — Descarga automática del CWRU Bearing Dataset.

El sitio oficial sirve cada archivo como `.mat` individual bajo una URL con el ID numérico:
    https://engineering.case.edu/sites/default/files/{N}.mat
Los IDs coinciden con las llaves de FILE_MAP en load_cwru.py (97.mat, 105.mat, ...).

Uso:
    from download_cwru import download_cwru
    from load_cwru import FILE_MAP
    report = download_cwru("raw", FILE_MAP)        # baja a raw/ los que falten
    print(report["ok"], report["skip"], report["fail"])

Estrategia: descarga del sitio oficial; si un archivo falla, se registra en report["fail"]
sin abortar el resto (el notebook avisa y sugiere fallback manual).
"""
from __future__ import annotations
import os
import time
import shutil
import urllib.request

OFFICIAL_URL = "https://engineering.case.edu/sites/default/files/{name}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (FL-FDD-V2 research; CWRU loader)"}


def download_one(url: str, dst: str, timeout: int = 120) -> int:
    """Descarga un archivo y devuelve su tamaño en bytes. Lanza excepción si falla/queda vacío.

    Hace streaming a disco (en vez de resp.read() completo en memoria) para reducir los
    IncompleteRead del servidor de Case, que trunca respuestas grandes con cierta frecuencia.
    """
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dst, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)  # bloques de 1 MB
        expected = resp.headers.get("Content-Length")
    size = os.path.getsize(dst)
    if size == 0:
        raise IOError("archivo descargado quedó vacío")
    if expected is not None and size != int(expected):
        raise IOError(f"descarga truncada: {size}/{expected} bytes")
    return size


def download_cwru(raw_dir: str, file_map: dict, base_url: str = OFFICIAL_URL,
                  timeout: int = 120, overwrite: bool = False, retries: int = 4) -> dict:
    """Descarga todos los .mat de file_map a raw_dir, con reintentos por archivo.

    Devuelve {'ok': [(name, size), ...], 'skip': [...], 'fail': [(name, motivo), ...]}.
    - skip: ya existía y no estaba vacío (no se vuelve a bajar salvo overwrite=True).
    - retries: nº de intentos por archivo ante errores transitorios (p.ej. IncompleteRead).
    """
    os.makedirs(raw_dir, exist_ok=True)
    report = {"ok": [], "skip": [], "fail": []}
    for name in file_map:
        dst = os.path.join(raw_dir, name)
        if os.path.exists(dst) and not overwrite and os.path.getsize(dst) > 0:
            report["skip"].append(name)
            continue
        last_err = None
        for intento in range(1, retries + 1):
            try:
                size = download_one(base_url.format(name=name), dst, timeout=timeout)
                report["ok"].append((name, size))
                last_err = None
                break
            except Exception as e:  # noqa: BLE001 - reintentar / seguir con el resto
                last_err = str(e)
                if os.path.exists(dst):
                    try:
                        os.remove(dst)
                    except OSError:
                        pass
                if intento < retries:
                    time.sleep(1.5 * intento)  # backoff lineal suave
        if last_err is not None:
            report["fail"].append((name, last_err))
    return report


if __name__ == "__main__":
    # Autodiagnóstico: baja a ./raw usando el FILE_MAP del cargador.
    from load_cwru import FILE_MAP  # type: ignore
    raw = os.path.join(os.path.dirname(__file__), "raw")
    rep = download_cwru(raw, FILE_MAP)
    print(f"OK={len(rep['ok'])}  SKIP={len(rep['skip'])}  FAIL={len(rep['fail'])}")
    for name, motivo in rep["fail"]:
        print(f"  FAIL {name}: {motivo}")
