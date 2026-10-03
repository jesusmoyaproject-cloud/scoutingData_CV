# Documentación de Integración: Headless Mode y JSON Tracking Dataset

## 📌 Resumen Técnico

Cuando `ENVIRONMENT = KAGGLE` o `HEADLESS_MODE = True`, el pipeline de **ScoutingData v5.0** opera en modo **Headless / High-Performance**:
- ❌ **No** genera archivos MP4 ni videos temporales.
- ❌ **No** renderiza minimapas ni dibuja overlays en OpenCV.
- ❌ **No** ejecuta `VideoWriter`.
- ✅ Se enfoca exclusivamente en: **Inferencia Directa (CUDA / OpenVINO) ➔ Tracking (ByteTrack) ➔ Homografía (2D Pitch) ➔ Exportación de `tracking_<video_name>.json`**.

---

## 📄 Contrato de Datos (`tracking_<video_name>.json`)

El archivo JSON guardado en `outputs/tracking_<video_name>.json` contiene el esquema de datos estandarizado:

```json
{
  "metadata": {
    "video_name": "match_01.mp4",
    "fps": 30.0,
    "total_frames": 1066,
    "duration_seconds": 35.53
  },
  "frames": [
    {
      "frame": 1,
      "timestamp": 0.033,
      "players": [
        {
          "track_id": 15,
          "pixel_x": 950.0,
          "pixel_y": 540.0,
          "pitch_x": 42.5,
          "pitch_y": 19.7
        }
      ],
      "ball": {
        "pixel_x": 952.0,
        "pixel_y": 550.0,
        "pitch_x": 43.1,
        "pitch_y": 20.0
      },
      "homography_valid": true
    }
  ],
  "events": []
}
```

---

## 📘 Guía de Integración para el Frontend y Analytics

### 1. Reconstrucción del Plano 2D (Minimap Interactivo)

El objeto `pitch_x` y `pitch_y` dentro de `players` y `ball` contiene las coordenadas normalizadas en metros sobre el terreno de juego real.

- **Dimensiones estándar del terreno:**
  - Largo ($X$): $0.0$ m a $105.0$ m (de portería izquierda a portería derecha).
  - Ancho ($Y$): $0.0$ m a $68.0$ m (de banda inferior a banda superior).
  
- **Renderizado en Canvas / SVG / WebGL:**
  Para dibujar el plano 2D interactivo en el cliente web:
  $$\text{canvas\_x} = \frac{\text{pitch\_x}}{105.0} \times \text{canvas\_width}$$
  $$\text{canvas\_y} = \frac{\text{pitch\_y}}{68.0} \times \text{canvas\_height}$$

- **Homografía No Válida (`homography_valid: false`):**
  Si en un frame `homography_valid` es `false`, significa que la cámara enfocó una toma cerrada (ej. primer plano o público) donde no hay suficientes líneas de cancha para proyectar en 2D. El Front puede ocultar o interpolar temporalmente las fichas del 2D durante esos frames.

---

### 2. Sincronización Video ↔ Frame ↔ Tracking

Para sincronizar la reproducción del video en el Frontend (HTML5 `<video>` o reproductores tipo Video.js / Shaka Player) con los datos del tracking:

1. **Obtener el tiempo actual del reproductor:**
   `const currentTime = videoElement.currentTime;`

2. **Calcular el índice del frame:**
   $$\text{frame\_idx} = \lfloor \text{currentTime} \times \text{fps} \rfloor + 1$$

3. **Búsqueda eficiente (O(1)):**
   Dado que los frames en `tracking.json` están ordenados secuencialmente de `1` a `total_frames`, la posición en el array de frames es exactamente `frame_idx - 1`:
   ```javascript
   const frameData = trackingData.frames[frame_idx - 1];
   if (frameData) {
     render2DPitch(frameData.players, frameData.ball);
   }
   ```

---

### 3. Incorporación de Eventos Etiquetados (`events`)

La clave `"events"` contiene una lista estructurada para almacenar eventos detectados de forma manual, semi-automática o por microservicios especializados.

- **Estructura futura de evento:**
  ```json
  {
    "frame": 530,
    "timestamp": 17.667,
    "type": "PASS",
    "player_from": 15,
    "player_to": 8,
    "confidence": 0.92
  }
  ```
- **Tipos de eventos soportados:** `PASS`, `SHOT`, `TACKLE`, `INTERCEPTION`, `FOUL`, `GOAL`.
- **Integración con Timeline:** El cliente web puede mapear la lista `events` directamente sobre la barra de tiempo (`timeline`) para habilitar navegación por marcadores ("Jump to pass at 00:17").

---

### 4. Mapeo `track_id` ↔ Jugador Real

El pipeline de vision asigna un identificador numérico persistente `track_id` (vía ByteTrack) a cada silueta detectada en el video.

Para asociar un `track_id` con el nombre, dorsal y foto del jugador real:

- **Estructura de Diccionario de Plantilla (Roster Mapping):**
  ```json
  {
    "mapping": {
      "15": {
        "player_id": "PL-0015",
        "name": "Lionel Messi",
        "number": 10,
        "team": "Home"
      },
      "8": {
        "player_id": "PL-0008",
        "name": "Pedri",
        "number": 8,
        "team": "Away"
      }
    }
  }
  ```
- **Flujo en la Interfaz de Usuario:**
  1. El analista o usuario selecciona una ficha en el minimap 2D (con `track_id: 15`).
  2. El frontend consulta la tabla de mapeo y despliega el perfil de "Lionel Messi".
  3. Si la asignación automática requiere corrección, la UI permite re-mapear un `track_id` a otro jugador de la plantilla sin necesidad de volver a procesar la inferencia del video.
