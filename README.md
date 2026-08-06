<div align="center">
  <h1>MeetMind</h1>
  <p><strong>A real-time AI meeting co-pilot powered by Gemini Live API</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Gemini-Live%20API-4285F4?style=flat-square&logo=google" />
    <img src="https://img.shields.io/badge/Google-ADK-34A853?style=flat-square&logo=google" />
    <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi" />
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python" />
  </p>

  <p><em>"The smartest voice in the room is always yours."</em></p>
</div>

---

## What is this?

You are in a meeting. Someone asks a question you should know the answer to. You do not know it. You cannot Google it without looking distracted.

MeetMind sits in a browser tab while you attend meetings. Whisper a question, get an instant answer through your earphones. Nobody knows.

---

## Features

- Real-time voice AI via Gemini Live API
- Private audio responses through your speakers only
- Screen awareness - share your screen and ask what you see
- Text input for when you cannot speak out loud
- Voice Activity Detection - AI stops the moment you speak
- Live conversation transcript

---
## Live demo: 

https://meetmind-671715875630.us-central1.run.app

## Screenshots

**1. Landing Page** - the homepage before starting a session.
![Landing](screenshots/landing%20page.png)

**2. Live Session Active** - Connected status and Listening... with animated wave bars.
![Live](screenshots/live%20session%20active.png)

**3. Voice Transcripts Working** - user speech and agent response in real time.
![Voice](screenshots/voice%20transcript%20working.png)

**4. Screen Share in Action** - MeetMind describing your screen when asked.
![Screen](screenshots/share%20screen%20in%20action.png)

**5. Typing a Question** - silent text input with instant AI response.
![Typing](screenshots/typing%20a%20ques.png)

**6. Screen Awareness** - asking what is visible and getting a grounded answer.
![Context](screenshots/github%20screenshare.png)

---

## Architecture

```
+-------------------------------------------------------------+
|                        Browser                              |
|                                                             |
|   Microphone  -->  Web Audio API  -->  PCM 16kHz chunks     |
|   Screen      -->  ImageCapture   -->  JPEG frames          |
|   Text Input  -->  WebSocket Client                         |
|                          |                                  |
|   Audio Output  <--  AudioContext (24kHz scheduled queue)   |
|   Transcript    <--  WebSocket Messages                     |
+--------------------------|----------------------------------+
                           |  WebSocket
                           |  audio / text / screen
                           |
+--------------------------|----------------------------------+
|                     FastAPI Server                         |
|                                                             |
|   upstream_task    browser messages --> LiveRequestQueue    |
|   downstream_task  Gemini events    --> WebSocket           |
|                                                             |
|   asyncio.gather(upstream, downstream)                      |
+--------------------------|----------------------------------+
                           |  Google ADK / StreamingMode.BIDI
                           |
+--------------------------|----------------------------------+
|                   Gemini Live API                          |
|                                                             |
|   Input:  audio/pcm 16kHz + image/jpeg + text              |
|   Output: audio/pcm 24kHz + transcriptions                 |
+-------------------------------------------------------------+
```

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JS, Web Audio API, Canvas |
| Backend | Python, FastAPI, asyncio WebSockets |
| AI | Google Gemini 2.0 Flash Live |
| Agent SDK | Google ADK, LiveRequestQueue |
| Audio In | PCM 16kHz, ScriptProcessorNode |
| Audio Out | PCM 24kHz, AudioBufferSourceNode |

---

## Project Structure

```
meetmind/
├── README.md
├── screenshots/
│
└── app/
    ├── main.py                  # FastAPI entry point, route definitions
    ├── requirements.txt
    ├── Dockerfile
    ├── pytest.ini
    │
    ├── core/
    │   ├── config.py            # env vars, audio and screen constants
    │   ├── session.py           # ADK runner, session creation, RunConfig
    │   └── pipeline.py          # upstream_task, downstream_task
    │
    ├── meetmind_agent/
    │   ├── agent.py             # Gemini Live agent, system prompt
    │   └── __init__.py
    │
    ├── static/
    │   └── index.html           # full frontend -- UI, audio, WebSocket
    │
    └── tests/
        └── test_meetmind.py     # pytest suite
```

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Google API key with Gemini Live access
- Chrome browser

### Setup

```bash
git clone https://github.com/pyarchana/meetmind.git
cd meetmind/app

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

cp .env.example .env
# add your GOOGLE_API_KEY to .env

uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in Chrome.

### Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## How It Works

### Voice Activity Detection
A `ScriptProcessorNode` monitors audio energy (RMS) in real time. When energy spikes across 3 consecutive frames, the AI is interrupted immediately, ensuring your voice always takes priority.

### Audio Scheduling
Incoming 24kHz PCM chunks from Gemini are decoded and queued sequentially using `AudioBufferSourceNode.start(when)` with a rolling `nextStartTime` cursor. This prevents gaps and overlaps, delivering seamless playback.

### Bidirectional Streaming
The FastAPI backend runs two concurrent asyncio tasks per WebSocket connection:
- `upstream_task` - Consumes browser messages (audio blobs, text input, screen frames) and pushes to `LiveRequestQueue`
- `downstream_task` - Iterates through `runner.run_live()` and forwards audio chunks and transcription events back to the browser

### Screen Context
Screen frames are captured every 5 seconds at 1280x720 resolution (JPEG quality 0.4) via `ImageCapture.grabFrame()` and sent as realtime blobs. The agent references screen content only when you explicitly ask.

---

## License

MIT

---

Built for the Gemini Live Agent Hackathon by [pyarchana](https://github.com/pyarchana), 2026.
