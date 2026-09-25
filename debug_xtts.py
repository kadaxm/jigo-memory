import os
import traceback

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.chdir(r"C:\Users\kadam\Documents\Jigo-memory")
from TTS.api import TTS  # noqa: E402

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
for name, text in [
    ("gu0016_roman", "Mane hackathon ma pahelu sthan! 48 kalak no-sleep, chhata maro demo judges ne sauthi gamyo!"),
    ("gu0001_script", "મમ્મી, મને placement મળી ગઈ! તમે કહ્યું હતું ને hard work કદી waste નઈ જાય."),
]:
    try:
        wav = tts.tts(text=name.split("_")[1] and text, speaker_wav="voice_sample.wav", language="hi")
        print(name, "OK samples:", len(wav), flush=True)
    except Exception:
        print(name, "FAILED:", flush=True)
        traceback.print_exc()