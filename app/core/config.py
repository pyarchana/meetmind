"""
Application configuration loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "meetmind"

GEMINI_MODEL = os.getenv(
    "MEETMIND_MODEL",
    "gemini-2.5-flash-native-audio-preview-09-2025"
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Audio settings
INPUT_SAMPLE_RATE = 16000   # Hz, browser captures at this rate
OUTPUT_SAMPLE_RATE = 24000  # Hz, Gemini responds at this rate

# Screen capture settings
SCREEN_JPEG_QUALITY = 0.4
SCREEN_CAPTURE_WIDTH = 1280
SCREEN_CAPTURE_HEIGHT = 720
SCREEN_CAPTURE_INTERVAL_MS = 5000
