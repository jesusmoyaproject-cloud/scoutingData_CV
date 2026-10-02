"""
ScoutingData v5.0 - Lanzador de Microservicios SOA (modo legacy / distribuido)

En v5.0 el modo por defecto es inferencia directa (sin HTTP).
Este script se usa solo si se quiere mantener la arquitectura SOA original
para despliegue distribuido o debugging por servicio.
Solo levanta 3 servicios (sin event_service).
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ServicesLauncher")

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import settings

SERVICES = [
    ("keypoint_service", 8001, "services.keypoint_service.main:app"),
    ("player_service",   8002, "services.player_service.main:app"),
    ("ball_service",     8003, "services.ball_service.main:app"),
    # event_service eliminado en v5.0
]

def kill_port_owner(port: int):
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
            for line in out.splitlines():
                if "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    else:
        try:
            subprocess.run(f"fuser -k {port}/tcp", shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def check_service_health(name: str, port: int, max_retries: int = 20) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(max_retries):
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code == 200:
                logger.info(f"✅ {name} (:{port}) OK → {r.json().get('service')}")
                return True
        except Exception:
            pass
        time.sleep(1)
    logger.error(f"❌ {name} (:{port}) no respondió.")
    return False

def main():
    logger.info(f"=== MICROSERVICIOS SOA v5.0 (Entorno: {settings.ENVIRONMENT}) ===")
    logger.info("⚠️  Modo legacy: en v5.0 la inferencia directa no requiere estos servicios.")

    for _, port, _ in SERVICES:
        kill_port_owner(port)
    time.sleep(1)

    processes = []
    for name, port, app_str in SERVICES:
        cmd = [sys.executable, "-m", "uvicorn", app_str,
               "--host", "127.0.0.1", "--port", str(port)]
        logger.info(f"Iniciando {name} en http://127.0.0.1:{port} ...")
        proc = subprocess.Popen(cmd, cwd=str(APP_DIR))
        processes.append((name, port, proc))

    all_ok = all(check_service_health(n, p) for n, p, _ in processes)
    logger.info("=" * 55)
    logger.info("3 microservicios OK" if all_ok else "Algunos servicios fallaron.")
    logger.info("Ctrl+C para detener.")
    logger.info("=" * 55)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nDeteniendo servicios...")
        for name, port, proc in processes:
            proc.terminate()
            logger.info(f"  {name} (:{port}) detenido.")
        sys.exit(0)

if __name__ == "__main__":
    main()
