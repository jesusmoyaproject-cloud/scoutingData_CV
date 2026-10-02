FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY services/ball_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ball_service /app/services/ball_service
COPY services/schemas.py /app/services/schemas.py

ENV MODEL_PATH=/app/model/openVino/modelBall_openvino_model
EXPOSE 8003

CMD ["uvicorn", "services.ball_service.main:app", "--host", "0.0.0.0", "--port", "8003"]
