# main.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
import requests
import json
import logging
from datetime import datetime
import uuid
from deepgram import DeepgramClient, PrerecordedOptions
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de logging detallado para backend
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(title="HeyGen Streaming API", version="1.0.0")

# Configurar CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins == "*":
    origins_list = ["*"]
else:
    origins_list = [origin.strip() for origin in allowed_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de API Keys desde variables de entorno
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
HEYGEN_BASE_URL = os.getenv("HEYGEN_BASE_URL", "https://api.heygen.com/v1")

# Configuración Deepgram
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Configuración OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_SYSTEM_MESSAGE = os.getenv("OPENAI_SYSTEM_MESSAGE")

# Variables globales para configuración dinámica
current_openai_key = OPENAI_API_KEY
current_system_message = OPENAI_SYSTEM_MESSAGE
openai_client = None

# Almacenamiento de sesiones activas
active_sessions: Dict[str, dict] = {}

# Modelos Pydantic
class SessionConfig(BaseModel):
    quality: str = Field(default_factory=lambda: os.getenv("SESSION_QUALITY", "medium"))
    avatar_id: str = Field(default_factory=lambda: os.getenv("AVATAR_ID", "75aa3befc98649d0bacd1a7266b1cfa3"))
    voice_id: str = Field(default_factory=lambda: os.getenv("VOICE_ID", "b03cee81247e42d391cecc6b60f0f042"))
    video_encoding: str = Field(default_factory=lambda: os.getenv("VIDEO_ENCODING", "H264"))

    version: str = Field(default_factory=lambda: os.getenv("SESSION_VERSION", "v2"))
    knowledge_base_id: str = Field(default_factory=lambda: os.getenv("KNOWLEDGE_BASE_ID", "197b84d8f4534ba68b0408bdaac78947"))

class TaskRequest(BaseModel):
    text: str
    task_type: str = "chat"  # "chat" o "repeat"

class SessionResponse(BaseModel):
    session_id: str
    status: str
    livekit_url: str = Field(..., alias="url")
    livekit_token: str = Field(..., alias="access_token")
    
class TaskResponse(BaseModel):
    task_id: str
    session_id: str
    status: str

class STTResponse(BaseModel):
    transcription: str
    confidence: float
    duration: float

# Clase para manejar sesiones de HeyGen
class HeyGenSessionManager:
    def __init__(self):
        if not HEYGEN_API_KEY:
            raise ValueError("HEYGEN_API_KEY environment variable is required")
        self.api_key_headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": HEYGEN_API_KEY
        }
        self.session_token: Optional[str] = None

    async def _get_session_token(self):
        """Obtiene un token de sesión temporal de la API de HeyGen."""
        if self.session_token:
            return
        url = f"{HEYGEN_BASE_URL}/streaming.create_token"
        logger.info("Obteniendo nuevo token de sesión de HeyGen...")
        try:
            response = requests.post(url, headers=self.api_key_headers)
            response.raise_for_status()
            data = response.json().get('data', {})
            self.session_token = data.get('token')
            if not self.session_token:
                raise HTTPException(status_code=500, detail="Failed to retrieve session token from HeyGen.")
            logger.info("Token de sesión obtenido con éxito.")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error getting session token: {str(e)}")

    async def _get_auth_headers(self) -> dict:
        """Asegura que hay un token y devuelve los headers de autorización."""
        await self._get_session_token()
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": HEYGEN_API_KEY,
            "authorization": f"Bearer {self.session_token}"
        }

    async def create_session(self, config: SessionConfig) -> dict:
        """Crea una nueva sesión en HeyGen y devuelve los datos, incluyendo credenciales de LiveKit."""
        auth_headers = await self._get_auth_headers()
        url = f"{HEYGEN_BASE_URL}/streaming.new"
        payload = {
            "quality": config.quality,
            "avatar_id": config.avatar_id,
            "voice": {"voice_id": config.voice_id, "rate": 1.1},
            "version": config.version,
            "knowledge_base_id": config.knowledge_base_id, #adding context
            "video_encoding": config.video_encoding
        }
        try:
            response = requests.post(url, json=payload, headers=auth_headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

    async def start_session(self, session_id: str) -> dict:
        """Inicia una sesión creada."""
        auth_headers = await self._get_auth_headers()
        url = f"{HEYGEN_BASE_URL}/streaming.start"
        payload = {"session_id": session_id}
        try:
            response = requests.post(url, json=payload, headers=auth_headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

    async def send_task(self, session_id: str, text: str, task_type: str) -> dict:
        """Envía una tarea a la sesión activa."""
        auth_headers = await self._get_auth_headers()
        url = f"{HEYGEN_BASE_URL}/streaming.task"
        payload = {
            "session_id": session_id,
            "text": text,
            "task_type": task_type
        }
        try:
            logger.debug(f"Enviando tarea a HeyGen: {payload}")
            response = requests.post(url, json=payload, headers=auth_headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error enviando tarea a HeyGen: {str(e)}")
            logger.error(f"Payload enviado: {payload}")
            logger.error(f"Status code: {getattr(e.response, 'status_code', 'N/A')}")
            if hasattr(e.response, 'text'):
                logger.error(f"Respuesta HeyGen: {e.response.text}")
            raise HTTPException(status_code=500, detail=f"Error sending task: {str(e)}")

    async def close_session(self, session_id: str) -> dict:
        """Cierra una sesión activa."""
        auth_headers = await self._get_auth_headers()
        url = f"{HEYGEN_BASE_URL}/streaming.stop"
        payload = {"session_id": session_id}
        try:
            response = requests.post(url, json=payload, headers=auth_headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error closing session: {str(e)}")

session_manager = HeyGenSessionManager()

# Función para procesar texto con OpenAI
async def process_with_openai(user_input: str) -> str:
    """
    Procesa el input del usuario con OpenAI GPT-5-nano optimizado para máxima velocidad.
    """
    global openai_client, current_openai_key, current_system_message
    
    if not current_openai_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")
    
    try:
        # Inicializar cliente si no existe o si cambió la API key
        if openai_client is None or openai_client.api_key != current_openai_key:
            openai_client = OpenAI(api_key=current_openai_key)
        
        # Intentar usar la nueva API de GPT-5 con parámetros de velocidad
        try:
            response = openai_client.responses.create(
                model="gpt-5-nano",
                input=[
                    {"role": "system", "content": current_system_message},
                    {"role": "user", "content": user_input}
                ],
                reasoning={
                    "effort": "minimal"  # Máxima velocidad, mínimo razonamiento
                },
                text={
                    "verbosity": "low"   # Respuestas concisas
                }
            )
            # Acceder al texto de respuesta según la documentación de GPT-5 nano
            response_text = ""
            if hasattr(response, 'output_text'):
                response_text = response.output_text.strip()
            elif hasattr(response, 'text'):
                response_text = response.text.strip()
            else:
                logger.warning(f"Estructura de respuesta desconocida: {type(response)}")
                response_text = str(response).strip()

            # Validar que el contenido no esté vacío
            if not response_text:
                logger.error("La respuesta de OpenAI está vacía")
                return "Lo siento, no pude generar una respuesta en este momento."

            return response_text
            
        except Exception as gpt5_error:
            logger.warning(f"Error con nueva API GPT-5, usando fallback: {str(gpt5_error)}")
            logger.debug(f"Tipo de error GPT-5: {type(gpt5_error).__name__}")
            
            # Fallback a la API tradicional de chat completions
            response = openai_client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": current_system_message},
                    {"role": "user", "content": user_input}
                ],
                max_completion_tokens=500       # Enfocar en tokens más probables
            )
            response_text = response.choices[0].message.content.strip()

            # Validar que el contenido no esté vacío
            if not response_text:
                logger.error("La respuesta de OpenAI (fallback) está vacía")
                return "Lo siento, no pude generar una respuesta en este momento."

            return response_text
        
    except Exception as e:
        logger.error(f"Error processing with OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing with OpenAI: {str(e)}")


# Endpoints REST

@app.get("/")
async def root():
    """Sirve la aplicación principal"""
    return FileResponse("avatar.html")

@app.get("/health")
async def health():
    """Endpoint de salud"""
    return {
        "status": "healthy",
        "service": "HeyGen Streaming API",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions)
    }

@app.post("/api/sessions/create", response_model=SessionResponse)
async def create_new_session(config: SessionConfig = SessionConfig()):
    """
    Crea una sesión, la inicia y devuelve las credenciales de LiveKit.
    """
    try:
        # 1. Crear la sesión en HeyGen
        create_response = await session_manager.create_session(config)
        session_data = create_response.get('data')
        if not session_data or 'session_id' not in session_data:
            raise HTTPException(status_code=500, detail="Respuesta inválida al crear sesión en HeyGen.")

        session_id = session_data['session_id']
        logger.info(f"[TÉCNICO] Sesión creada en HeyGen: {session_id}")

        # 2. Iniciar la sesión
        await session_manager.start_session(session_id)
        logger.info(f"[TÉCNICO] Sesión iniciada en HeyGen: {session_id}")
        
        # 3. Almacenar localmente y devolver credenciales
        session_created_time = datetime.now()
        active_sessions[session_id] = {
            "session_id": session_id,
            "status": "active",
            "created_at": session_created_time.isoformat(),
            "livekit_url": session_data.get("url"),
            "livekit_token": session_data.get("access_token")
        }

        logger.info(f"[DEBUG] Session {session_id} stored in active_sessions at {session_created_time.isoformat()}")
        logger.info(f"[DEBUG] Total active sessions after storage: {len(active_sessions)}")
        logger.info(f"[DEBUG] All session IDs after storage: {list(active_sessions.keys())}")
        
        return SessionResponse(
            session_id=session_id,
            status="active",
            url=session_data.get("url"),
            access_token=session_data.get("access_token")
        )
    except Exception as e:
        logger.error(f"Fallo en el flujo de creación de sesión: {e}")
        raise e

@app.post("/api/sessions/{session_id}/task")
async def send_session_task(session_id: str, task: TaskRequest):
    """Envía una tarea de texto a una sesión activa."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    response = await session_manager.send_task(session_id, task.text, task.task_type)
    return {"status": "task_sent", "response": response}

@app.delete("/api/sessions/{session_id}")
async def close_heygen_session(session_id: str):
    """Cierra una sesión activa en HeyGen. Idempotente - no falla si la sesión ya fue cerrada."""
    # Verificar si la sesión existe
    if session_id in active_sessions:
        # Intentar cerrar en HeyGen
        try:
            await session_manager.close_session(session_id)
            logger.info(f"[TÉCNICO] Sesión cerrada en HeyGen: {session_id}")
        except Exception as e:
            logger.warning(f"[TÉCNICO] Error cerrando sesión en HeyGen (puede ya estar cerrada): {e}")

        # Eliminar de sesiones activas
        del active_sessions[session_id]
        logger.info(f"[TÉCNICO] Sesión eliminada de active_sessions: {session_id}")
    else:
        logger.info(f"[TÉCNICO] Sesión {session_id} ya fue cerrada previamente (idempotente)")

    return {"status": "closed", "session_id": session_id}

@app.post("/api/stt/transcribe", response_model=STTResponse)
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Transcribe un archivo de audio usando Deepgram STT.
    """
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="Deepgram API key not configured")
    
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Verificar tipo de archivo
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg"]
    if audio_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Leer el archivo de audio
        audio_data = await audio_file.read()
        
        # Crear cliente de Deepgram
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        
        # Configurar opciones para la transcripción
        options = PrerecordedOptions(
            model="nova-2",
            language="es",  # Español
            smart_format=True,
            punctuate=True,
            diarize=False
        )
        
        # Crear payload para transcripción
        payload = {
            "buffer": audio_data,
        }
        
        # Transcribir audio usando el método correcto
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        
        # Extraer resultado
        transcript = response["results"]["channels"][0]["alternatives"][0]
        transcription = transcript["transcript"]
        confidence = transcript["confidence"]
        
        # Obtener duración del audio
        duration = response["metadata"]["duration"]
        
        if not transcription.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        logger.info(f"[STT] Audio transcrito exitosamente (confianza: {confidence:.2f}): '{transcription[:50]}...'")

        return STTResponse(
            transcription=transcription,
            confidence=confidence,
            duration=duration
        )
        
    except Exception as e:
        logger.error(f"Error en transcripción de audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint para manejar la señalización WebRTC con HeyGen.
    El frontend se conecta aquí para recibir la información de la sesión
    y establecer la conexión WebRTC directamente con HeyGen.
    """
    import asyncio

    # Logging detallado para debug con timestamp
    websocket_attempt_time = datetime.now()
    logger.info(f"[DEBUG] WebSocket connection attempt for session: {session_id} at {websocket_attempt_time.isoformat()}")
    logger.info(f"[DEBUG] Active sessions count: {len(active_sessions)}")
    logger.info(f"[DEBUG] Active session IDs: {list(active_sessions.keys())}")

    # Si la sesión existe, mostrar el timing de creación vs conexión
    if session_id in active_sessions:
        session_data = active_sessions[session_id]
        created_at_str = session_data.get("created_at", "unknown")
        logger.info(f"[DEBUG] Session {session_id} was created at: {created_at_str}")
        logger.info(f"[DEBUG] WebSocket connecting at: {websocket_attempt_time.isoformat()}")
    else:
        logger.info(f"[DEBUG] Session {session_id} not found in active_sessions during initial check")

    # Implementar mecanismo de reintento para manejar condición de carrera
    max_retries = 5
    retry_delay = 0.5  # 500ms inicial

    session_found = False
    session_valid = False

    for attempt in range(max_retries):
        if session_id in active_sessions:
            # Verificar que la sesión esté en estado válido
            session_data = active_sessions[session_id]
            session_status = session_data.get("status", "unknown")
            livekit_url = session_data.get("livekit_url")
            livekit_token = session_data.get("livekit_token")

            if session_status == "active" and livekit_url and livekit_token:
                session_found = True
                session_valid = True
                logger.info(f"[DEBUG] Session {session_id} FOUND and VALID after {attempt + 1} attempts")
                logger.info(f"[DEBUG] Session status: {session_status}, has credentials: {bool(livekit_url and livekit_token)}")
                break
            else:
                logger.warning(f"[DEBUG] Session {session_id} found but INVALID - status: {session_status}, has_credentials: {bool(livekit_url and livekit_token)}")

        if attempt < max_retries - 1:  # No hacer delay en el último intento
            logger.info(f"[DEBUG] Session {session_id} not found or invalid, attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
            retry_delay *= 1.5  # Incremento exponencial del delay
        else:
            logger.error(f"[DEBUG] Session {session_id} NOT FOUND OR INVALID after {max_retries} attempts")
            logger.error(f"[DEBUG] Available sessions: {list(active_sessions.keys())}")

    if not session_found or not session_valid:
        error_reason = "Session not found" if not session_found else "Session invalid"
        logger.error(f"[DEBUG] WebSocket connection rejected: {error_reason}")
        await websocket.close(code=1008, reason=f"{error_reason} after retries")
        return
    
    await websocket.accept()
    logger.info(f"[TÉCNICO] WebSocket conectado para sesión: {session_id}")
    
    try:
        # Enviar información de la sesión inmediatamente después de conectar
        session_data = active_sessions[session_id]
        await websocket.send_text(json.dumps({
            "type": "session_info",
            "data": {
                "session_id": session_id,
                "livekit_url": session_data["livekit_url"],
                "livekit_token": session_data["livekit_token"]
            }
        }))
        
        # Manejar mensajes entrantes del frontend
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "task":
                    # Procesar tarea con OpenAI y enviar como "repeat" al avatar
                    user_input = message.get("text", "")
                    if user_input:
                        try:
                            logger.info(f"[CONVERSACIÓN] Usuario ({session_id[:8]}): {user_input}")

                            # Procesar con OpenAI
                            await websocket.send_text(json.dumps({
                                "type": "processing",
                                "message": "Procesando con OpenAI..."
                            }))

                            openai_response = await process_with_openai(user_input)
                            logger.info(f"[CONVERSACIÓN] Guardiana ({session_id[:8]}): {openai_response}")

                            # Enviar la respuesta de OpenAI como "repeat" al streaming
                            await session_manager.send_task(session_id, openai_response, "repeat")

                            await websocket.send_text(json.dumps({
                                "type": "task_sent",
                                "message": "Respuesta enviada al avatar",
                                "user_input": user_input,
                                "openai_response": openai_response
                            }))

                            logger.info(f"[TÉCNICO] Tarea completada exitosamente para sesión {session_id[:8]}")
                        except Exception as e:
                            logger.error(f"[TÉCNICO] Error procesando tarea para sesión {session_id[:8]}: {str(e)}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": f"Error procesando con OpenAI: {str(e)}"
                            }))
                
                elif message.get("type") == "close":
                    # Cerrar sesión (fallback - el método principal es el endpoint DELETE)
                    if session_id in active_sessions:
                        try:
                            await session_manager.close_session(session_id)
                            logger.info(f"[TÉCNICO] Sesión cerrada en HeyGen desde WebSocket: {session_id}")
                        except Exception as e:
                            logger.warning(f"[TÉCNICO] Error cerrando sesión en HeyGen desde WebSocket: {e}")
                        del active_sessions[session_id]
                        logger.info(f"[TÉCNICO] Sesión eliminada de active_sessions desde WebSocket: {session_id}")
                    else:
                        logger.info(f"[TÉCNICO] Sesión {session_id} ya fue cerrada (WebSocket)")
                    break
                    
            except WebSocketDisconnect:
                logger.info(f"[TÉCNICO] WebSocket desconectado para sesión: {session_id}")
                break
            except Exception as e:
                logger.error(f"[TÉCNICO] Error en WebSocket para sesión {session_id}: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                }))

    except WebSocketDisconnect:
        logger.info(f"[TÉCNICO] WebSocket desconectado para sesión: {session_id}")
    except Exception as e:
        logger.error(f"[TÉCNICO] Error general en WebSocket para sesión {session_id}: {e}")
    finally:
        logger.info(f"[TÉCNICO] Cerrando WebSocket para sesión: {session_id}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)