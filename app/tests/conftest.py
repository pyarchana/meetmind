"""
main.py refuses to import without GOOGLE_API_KEY, which is deliberate. The
integration tests import it, so give them a placeholder unless a real key is
already in the environment. The live test checks for this exact value to
decide whether it has a real key to work with.
"""

import os

PLACEHOLDER_KEY = "test-key-not-used"

os.environ.setdefault("GOOGLE_API_KEY", PLACEHOLDER_KEY)
