FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY services/keypoint_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/keypoint_service /app/services/keypoint_service
COPY services/schemas.py /app/services/schemas.py

ENV MODEL_PATH=/app/model/openVino/modelKeypoint_openvino_model
EXPOSE 8001

CMD ["uvicorn", "services.keypoint_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
