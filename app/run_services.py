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
    ("player_service", 8002, "services.player_service.main:app"),
    ("ball_service", 8003, "services.ball_service.main:app"),
    ("event_service", 8004, "services.event_service.main:app"),
]

def kill_port_owner(port: int):
    """
    Liberación portátil de puertos compatible con Windows y Linux (Kaggle).
    """
    if sys.platform == "win32":
        try:
            cmd = f'netstat -ano | findstr :{port}'
            output = subprocess.check_output(cmd, shell=True).decode()
            for line in output.splitlines():
                if 'LISTENING' in line:
                    pid = line.strip().split()[-1]
                    logger.info(f"Limpiando puerto {port} (proceso PID {pid})...")
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    else:
        try:
            subprocess.run(f"fuser -k {port}/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def check_service_health(name: str, port: int, max_retries: int = 20):
    url = f"http://127.0.0.1:{port}/health"
    for attempt in range(1, max_retries + 1):
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                logger.info(f"✅ {name} (Puerto {port}): OK -> {data.get('service')}")
                return True
        except Exception:
            pass
        time.sleep(1)
    logger.error(f"❌ {name} (Puerto {port}): No respondió tras {max_retries}s")
    return False

def main():
    logger.info(f"=== INICIANDO MICROSERVICIOS SOA (Entorno: {settings.ENVIRONMENT}) ===")
    
    # 1. Liberar puertos
    for _, port, _ in SERVICES:
        kill_port_owner(port)

    time.sleep(1)
    processes = []

    # 2. Iniciar servicios
    for name, port, app_str in SERVICES:
        cmd = [sys.executable, "-m", "uvicorn", app_str, "--host", "127.0.0.1", "--port", str(port)]
        logger.info(f"Iniciando {name} en http://127.0.0.1:{port} ...")
        proc = subprocess.Popen(cmd, cwd=str(APP_DIR))
        processes.append((name, port, proc))

    # 3. Validar Health Checks con Retry Loop mientras cargan modelos OpenVINO
    logger.info("Esperando inicialización de modelos OpenVINO en memoria...")
    all_ok = True
    for name, port, _ in processes:
        ok = check_service_health(name, port, max_retries=20)
        if not ok:
            all_ok = False

    logger.info("=======================================================")
    if all_ok:
        logger.info("¡Los 4 microservicios (Keypoint, Player, Ball, Event) están 100% activos!")
    else:
        logger.warning("Algunos servicios tardaron en responder. Revisa los logs arriba.")

    logger.info("Presiona Ctrl + C en esta terminal para detenerlos.")
    logger.info("=======================================================")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nDeteniendo todos los microservicios...")
        for name, port, proc in processes:
            proc.terminate()
            logger.info(f"{name} (Puerto {port}) detenido.")
        sys.exit(0)

if __name__ == "__main__":
    main()
