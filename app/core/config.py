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

# The browser captures at this rate and tags every audio blob with it.
INPUT_SAMPLE_RATE = 16000
