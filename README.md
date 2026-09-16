<div align="center">
  <h1>MeetMind</h1>
  <p><strong>A real-time AI meeting co-pilot powered by Gemini Live API</strong></p>

  <p>
    <a href="https://github.com/pyarchana/meetmind/actions/workflows/ci.yml"><img src="https://github.com/pyarchana/meetmind/actions/workflows/ci.yml/badge.svg" /></a>
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
| Frontend | React, Vite, Tailwind, Web Audio API |
| Backend | Python, FastAPI, asyncio WebSockets |
| AI | Google Gemini 2.5 Flash Native Audio |
| Agent SDK | Google ADK, LiveRequestQueue |
| Audio In | PCM 16kHz, ScriptProcessorNode |
| Audio Out | PCM 24kHz, AudioBufferSourceNode |

---

## Project Structure

```
meetmind/
├── README.md
├── deploy.sh                    # Cloud Run deployment
├── architecture.png
├── screenshots/
│
├── bench/
│   ├── latency.py               # end to end latency benchmark
│   ├── mock_server.py           # fake server for checking the benchmark
│   └── test_latency.py
│
├── web/                         # frontend source, builds into app/static
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── components/          # Board, Transcript, Composer, StatusPill
│       └── lib/                 # audio.js, session.js and their tests
│
└── app/
    ├── main.py                  # FastAPI entry point, route definitions
    ├── requirements.txt
    ├── Dockerfile
    ├── .dockerignore
    ├── .env.example
    ├── pytest.ini
    │
    ├── core/
    │   ├── config.py            # env vars, model and audio settings
    │   ├── session.py           # ADK runner, session creation, RunConfig
    │   └── pipeline.py          # upstream_task, downstream_task
    │
    ├── meetmind_agent/
    │   ├── agent.py             # Gemini Live agent, system prompt
    │   └── __init__.py
    │
    ├── static/                   # built frontend, generated by web/
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

Open `http://127.0.0.1:8000` in Chrome. The built frontend is committed, so
this works without installing node.

### Working on the frontend

```bash
cd web
npm install
npm run dev     # vite on :5173, proxies /ws to :8000
npm test
npm run build   # writes into app/static
```

`app/static` is generated. Commit the rebuild whenever you change anything in
`web/src`, or CI will flag the two as out of step.

### Running Tests

```bash
pip install -r requirements-dev.txt
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

## Performance

Measured rather than claimed. The harness lives in [bench/](bench/), and
[bench/README.md](bench/README.md) covers how to run it and how to read it.

Client side, fixed by the constants in `web/src/lib/audio.js` and pinned by the
web tests:

| What | Cost |
|---|---|
| Mic frame size | 64 ms, so no client timing is finer than that |
| Speech to playback cut | up to 192 ms, three frames of confirmation |
| Silence to end of turn | 1600 ms, twenty five frames |

Server round trip numbers need a run against a real key:

```bash
cd app && MEETMIND_TIMING=1 uvicorn main:app --port 8000
```

```bash
python bench/latency.py --runs 20 --label "local, wifi" --out bench/results/local.json
```

Those numbers are what the mic buffer was moved to 1024 samples for. It used
to be 4096, which put barge in at 768 ms, longer than the whole server round
trip on most networks, so the thing making interruptions feel slow was the
client rather than the model.

Measured offline, no key needed, in
[bench/results/audio_tradeoffs.json](bench/results/audio_tradeoffs.json):

| Frame size | Barge in floor | Wire | CPU per second |
|---|---|---|---|
| 1024 samples | 192 ms | 42.1 KB/s | 330 us |
| 4096 samples (today) | 768 ms | 41.8 KB/s | 271 us |

Going to 1024 sample frames cut the barge in floor four fold for 0.3 KB/s and
sixty microseconds of CPU, with no server change. The cost is sensitivity: a
transient now only needs 192 ms to trigger, so if false barge ins appear,
SPEECH_FRAMES is the dial.

Base64 over text frames costs 33.7 percent against binary, which is 10 KB/s on
a 42 KB/s stream and 68 microseconds per frame. Real, but not worth a protocol
change on its own.

## License

MIT

---

Built for the Gemini Live Agent Hackathon by [pyarchana](https://github.com/pyarchana), 2026.
