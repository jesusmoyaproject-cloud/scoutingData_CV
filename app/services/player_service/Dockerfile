FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY services/player_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/player_service /app/services/player_service
COPY services/schemas.py /app/services/schemas.py

ENV MODEL_PATH=/app/model/openVino/modelPlayer_openvino_model
EXPOSE 8002

CMD ["uvicorn", "services.player_service.main:app", "--host", "0.0.0.0", "--port", "8002"]
