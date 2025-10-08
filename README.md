# 🎬 Avatar AI - HeyGen Streaming Client

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![WebRTC](https://img.shields.io/badge/WebRTC-Enabled-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Avatar AI** es una aplicación de streaming de avatares que integra la API de HeyGen para crear y controlar avatares de IA en tiempo real a través de WebRTC. El sistema consiste en un backend FastAPI que gestiona las sesiones de HeyGen y sirve un cliente web para la interacción en tiempo real con avatares.

## ✨ Características principales

- 🤖 **Avatares IA en tiempo real** - Streaming de avatares usando HeyGen API
- 🎥 **Video streaming WebRTC** - Conexión directa de baja latencia
- 🎯 **Chroma key avanzado** - Procesamiento de fondo en tiempo real con controles personalizables
- 🔊 **Audio sincronizado** - Síntesis de voz integrada con avatares
- 🌐 **Interfaz web moderna** - Cliente web responsive con controles intuitivos
- ⚡ **FastAPI backend** - API REST y WebSocket para gestión de sesiones
- 🛠️ **Altamente configurable** - Personalización de avatares, voces y calidad de video

## 🏗️ Arquitectura

```
┌─────────────────┐    WebSocket    ┌─────────────────┐    HeyGen API    ┌─────────────────┐
│   Frontend      │◄────────────────┤   Backend       │◄─────────────────┤   HeyGen        │
│   (avatar.html) │                 │   (FastAPI)     │                  │   Services      │
│                 │                 │                 │                  │                 │
│ ┌─────────────┐ │                 │ ┌─────────────┐ │                  │ ┌─────────────┐ │
│ │ Video Stream│ │   WebRTC        │ │ Session Mgr │ │                  │ │ LiveKit     │ │
│ │ + Chroma Key│ │◄────────────────┼─┤ + Auth      │ │                  │ │ + Avatar AI │ │
│ │             │ │                 │ │             │ │                  │ │             │ │
│ └─────────────┘ │                 │ └─────────────┘ │                  │ └─────────────┘ │
└─────────────────┘                 └─────────────────┘                  └─────────────────┘
```

### Componentes principales

**Backend (main.py)**
- **FastAPI application** - Servidor REST API y WebSocket
- **HeyGenSessionManager** - Gestión de sesiones y autenticación con HeyGen
- **Session storage** - Almacenamiento en memoria de sesiones activas
- **CORS middleware** - Configuración para desarrollo

**Frontend (avatar.html)**
- **Cliente WebRTC** - Conexión directa a servidores LiveKit de HeyGen
- **Procesamiento de video** - Chroma key en tiempo real con Canvas API
- **UI responsiva** - Controles de sesión y configuración de video
- **Sistema de logs** - Monitoreo en tiempo real del estado

## 🚀 Instalación y configuración

### Requisitos previos

- Python 3.8+
- Navegador web moderno con soporte WebRTC
- Cuenta de HeyGen con API key activa

### 1. Clonar el repositorio

```bash
git clone https://github.com/EgasJulian/Avatar-AI.git
cd Avatar-AI/V2
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar API de HeyGen

Edita el archivo `main.py` y actualiza la configuración:

```python
# Línea 29: Reemplaza con tu API key de HeyGen
HEYGEN_API_KEY = "tu_api_key_aqui"

# Líneas 38-41: Configura avatar y voz
class SessionConfig(BaseModel):
    avatar_id: str = "Marianne_ProfessionalLook2_public"
    voice_id: str = "253dc1d148f2410a860bc28996b30621"
    quality: str = "medium"
    video_encoding: str = "H264"
```

### 5. Iniciar la aplicación

```bash
# Opción 1: Directamente con Python
python main.py

# Opción 2: Con Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Acceder al cliente

Abre `avatar.html` en tu navegador o usa un servidor web local:

```bash
# Con Python
python -m http.server 3000

# Luego visita: http://localhost:3000/avatar.html
```

## 🎮 Guía de uso

### Iniciar una sesión

1. **Crear sesión** - Click en "Crear Sesión" para inicializar el avatar
2. **Conexión automática** - El sistema se conecta automáticamente via WebSocket y WebRTC
3. **Video activo** - El avatar aparece con chroma key activado por defecto

### Enviar tareas al avatar

1. **Escribir texto** - Ingresa el texto en el área de "Enviar Tarea"
2. **Enviar** - Click en "Enviar Tarea" o Ctrl+Enter
3. **Observar** - El avatar pronunciará el texto en tiempo real

### Controles de video

- **🎬 Video Original** - Muestra el video sin procesamiento
- **🎯 Chroma Key** - Activa el procesamiento de fondo (modo por defecto)
- **Slider de tolerancia** - Ajusta la sensibilidad del chroma key

### Cerrar sesión

- Click en "Cerrar Sesión" para terminar la conexión y limpiar recursos

## 📡 API Reference

### REST Endpoints

#### Crear sesión
```http
POST /api/sessions/create
Content-Type: application/json

{
  "quality": "medium",
  "avatar_id": "Marianne_ProfessionalLook2_public",
  "voice_id": "253dc1d148f2410a860bc28996b30621"
}
```

**Respuesta:**
```json
{
  "session_id": "uuid4-string",
  "status": "created",
  "livekit_url": "wss://...",
  "livekit_token": "eyJhbG..."
}
```

#### Enviar tarea
```http
POST /api/sessions/{session_id}/task
Content-Type: application/json

{
  "text": "Hola, este es un mensaje de prueba",
  "task_type": "repeat"
}
```

#### Cerrar sesión
```http
DELETE /api/sessions/{session_id}
```

### WebSocket Events

#### Conexión
```javascript
ws = new WebSocket('ws://localhost:8000/ws/{session_id}');
```

#### Eventos enviados por el cliente
```javascript
// Enviar tarea
ws.send(JSON.stringify({
  type: 'task',
  text: 'Texto para el avatar'
}));

// Cerrar sesión
ws.send(JSON.stringify({
  type: 'close'
}));
```

#### Eventos recibidos del servidor
```javascript
// Información de sesión
{
  "type": "session_info",
  "data": {
    "livekit_url": "wss://...",
    "livekit_token": "eyJ..."
  }
}

// Error
{
  "type": "error",
  "message": "Descripción del error"
}
```

## ⚙️ Configuración avanzada

### Personalización del avatar

Puedes cambiar el avatar editando la clase `SessionConfig` en `main.py`:

```python
class SessionConfig(BaseModel):
    avatar_id: str = "tu_avatar_id"      # ID del avatar de HeyGen
    voice_id: str = "tu_voice_id"        # ID de la voz
    quality: str = "high"                # low, medium, high
    video_encoding: str = "H264"         # H264, VP8, VP9
```

### Configuración del chroma key

Los parámetros del chroma key se pueden ajustar en `avatar.html`:

```javascript
// Línea 583: Tolerancia por defecto
let tolerance = 30; // 10-100, mayor = más agresivo

// Líneas 1003-1007: Color de fondo de reemplazo
data[i] = Math.floor(15 + gradientFactor * 25);     // R
data[i + 1] = Math.floor(20 + gradientFactor * 35); // G  
data[i + 2] = Math.floor(45 + gradientFactor * 60); // B
```

### Variables de entorno

Para producción, se recomienda usar variables de entorno:

```bash
export HEYGEN_API_KEY="tu_api_key"
export PORT="8000"
export CORS_ORIGINS="https://tu-dominio.com"
```

## 🛠️ Desarrollo

### Estructura del proyecto

```
Avatar-AI/V2/
├── main.py              # Backend FastAPI
├── avatar.html          # Cliente web completo
├── requirements.txt     # Dependencias Python
├── CLAUDE.md           # Documentación para Claude
├── README.md           # Este archivo
└── __pycache__/        # Cache de Python
```

### Ejecutar en modo desarrollo

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Logs y debugging

El backend genera logs detallados:

```python
# Configurar nivel de logging
logging.basicConfig(level=logging.DEBUG)
```

El frontend incluye sistema de logs en tiempo real visible en la interfaz.

### Testing

Para probar la aplicación:

1. Verificar conectividad con HeyGen API
2. Probar creación de sesiones
3. Verificar streaming WebRTC
4. Testear funcionalidad de chroma key

## 🐛 Troubleshooting

### Problemas comunes

**Error: "Failed to create session"**
- Verifica tu API key de HeyGen
- Confirma que tu cuenta tiene créditos disponibles
- Revisa los logs del backend para más detalles

**Video no se muestra**
- Verifica que WebRTC esté habilitado en tu navegador
- Comprueba la conexión a internet
- Revisa la consola del navegador para errores

**Chroma key no funciona correctamente**
- Ajusta el slider de tolerancia
- Verifica que el navegador soporte Canvas API
- El avatar debe tener un fondo verde para mejor resultado

**Audio no se reproduce**
- Haz click en el video para activar audio (requerimiento del navegador)
- Verifica configuración de audio del navegador
- Comprueba que el avatar tenga configuración de voz

### Logs útiles

**Backend (FastAPI):**
```bash
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Session created: uuid-session-id
INFO:     WebSocket connected: uuid-session-id
```

**Frontend (Consola del navegador):**
```javascript
[timestamp] ✅ Sesión creada con éxito: uuid
[timestamp] ✅ WebSocket conectado.
[timestamp] ✅ Conectado a LiveKit
```

## 📦 Dependencias

### Backend
- `fastapi` - Framework web moderno
- `uvicorn[standard]` - Servidor ASGI
- `websockets` - Soporte WebSocket
- `requests` - Cliente HTTP
- `pydantic` - Validación de datos
- `python-multipart` - Manejo de formularios
- `aiofiles` - Operaciones de archivos asíncronas

### Frontend
- `LiveKit Client SDK` - Cliente WebRTC (cargado via CDN)
- Canvas API (nativo del navegador)
- WebSocket API (nativo del navegador)

## 🤝 Contribución

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👥 Créditos

- **HeyGen** - Plataforma de avatares IA y API
- **LiveKit** - Infraestructura WebRTC
- **FastAPI** - Framework web Python

## 📞 Soporte

Para soporte y preguntas:

- 📧 Email: [tu-email@dominio.com]
- 🐛 Issues: [GitHub Issues](https://github.com/EgasJulian/Avatar-AI/issues)
- 📖 Documentación: [Wiki del proyecto](https://github.com/EgasJulian/Avatar-AI/wiki)

---

**⚠️ Nota importante:** Este proyecto requiere una cuenta activa de HeyGen y su correspondiente API key. Los costos de uso de la API de HeyGen corren por cuenta del usuario.

---

<p align="center">
  Hecho con ❤️ para democratizar el acceso a avatares IA
</p>