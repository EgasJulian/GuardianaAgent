# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an avatar streaming application that integrates with HeyGen's streaming API to create and control AI avatars via WebRTC. The system consists of a FastAPI backend that manages HeyGen sessions and serves a web client for real-time avatar interaction.

## Architecture

### Backend (main.py)
- **FastAPI application** serving as a bridge between the frontend and HeyGen's streaming API
- **HeyGenSessionManager class** handles all HeyGen API interactions including:
  - Session token management and authentication
  - Session creation, starting, and termination
  - Task dispatch (text-to-speech/avatar actions)
- **REST API endpoints** for session management (`/api/sessions/create`, `/api/sessions/{id}/task`, etc.)
- **Session storage** maintains active sessions in memory with LiveKit credentials

### Frontend (avatar.html)
- **Single-page web client** with WebRTC integration for real-time avatar streaming
- **Dual connection model**: WebSocket to backend + WebRTC to HeyGen's LiveKit servers
- **Session management UI** with connection status, controls, and task submission
- **Video streaming** displays the avatar video feed via WebRTC

### Key Integration Points
- Backend creates HeyGen sessions and returns LiveKit credentials to frontend
- Frontend establishes direct WebRTC connection to HeyGen using provided credentials
- Text tasks are sent via REST API to backend, which forwards them to HeyGen

## Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend server
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000

# Open the frontend
# Open avatar.html in a web browser or serve it via a local server
```

### Configuration
- **HeyGen API Key**: Set in `main.py:29` (currently hardcoded - consider environment variables for production)
- **Avatar/Voice IDs**: Configured in `SessionConfig` class (`main.py:36-41`)
- **CORS settings**: Currently allows all origins (`main.py:20-26`)

## File Structure
- `main.py` - FastAPI backend server with HeyGen integration
- `avatar.html` - Complete frontend client with WebRTC and UI
- `requirements.txt` - Python dependencies
- `__pycache__/` - Python bytecode cache (auto-generated)

## API Integration Notes
- Uses HeyGen's v1 streaming API with token-based authentication
- Requires session token before creating streaming sessions
- Sessions must be explicitly started after creation
- WebRTC connection is direct from client to HeyGen's LiveKit servers

## Development Notes
- Backend runs on port 8000 by default
- Frontend expects backend at `http://localhost:8000`
- No database - sessions stored in memory (will be lost on restart)
- CORS configured for development (all origins allowed)