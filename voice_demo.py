import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import save

load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Generate speech using a built-in voice (no reference clip needed)
audio = client.text_to_speech.convert(
    voice_id="JBFqnCBsd6RMkjVDRZzb",  # "George" - a built-in ElevenLabs voice
    text="Hello, I am an AI voice demo built using ElevenLabs. My name is Jigo, and I can speak in any voice you give me.",
    model_id="eleven_multilingual_v2"
)

save(audio, "output.wav")
print("Done. Check output.wav in your folder.")