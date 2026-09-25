"""Stage 1.5 smoke test: rumik-oss-1 in 4-bit on RTX 3050.
Three lines: A) Ira-sad Hindi baseline  B) male-desc + Gujarati  C) male-desc + Hinglish.
Outputs WAVs + measures VRAM and generation speed."""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel, BitsAndBytesConfig

REPO = "rumik-ai/rumik-oss-1"
OUT = r"C:\Users\kadam\Documents\Jigo-memory\assets\rumik_samples\generated"
os.makedirs(OUT, exist_ok=True)

print("loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)

print("loading model in 4-bit (bitsandbytes)... first load is the slow part", flush=True)
t0 = time.time()
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    REPO, trust_remote_code=True, dtype=torch.bfloat16,
    quantization_config=bnb, device_map={"": 0},
).eval()
load_s = time.time() - t0
print(f"model loaded in {load_s:.1f}s", flush=True)

print("loading mimi codec...", flush=True)
mimi = MimiModel.from_pretrained(REPO, subfolder="codec").eval().cuda()
print("mimi loaded", flush=True)

torch.cuda.reset_peak_memory_stats()
vram_gb = torch.cuda.max_memory_allocated() / 1e9
reserved_gb = torch.cuda.memory_reserved() / 1e9
print(f"VRAM allocated: {vram_gb:.2f} GB | reserved: {reserved_gb:.2f} GB", flush=True)

LINES = [
    ("A_ira_sad_hindi_baseline", "Ira", "sad, Hindi accent, slow pace",
     "\u092e\u093e\u0901, \u0906\u091c \u092b\u093f\u0930 \u0924\u0941\u092e\u094d\u0939\u093e\u0930\u0947 \u0932\u093f\u090f \u091a\u093e\u092f \u092c\u0928\u093e \u0926\u0940\u0964 \u0915\u092a \u0920\u0902\u0921\u093e \u0939\u094b \u0917\u092f\u093e, \u092a\u0930 \u0924\u0941\u092e\u094d\u0939\u093e\u0930\u093e \u0907\u0902\u0924\u091c\u093c\u093e\u0930 \u0928\u0939\u0940\u0902\u0964"),
    ("B_male_desc_gujarati", "Ira",
     "a deep male voice, middle-aged Gujarati man, unhurried drawl, warm, thick accent, completely unfazed",
     "\u0a9a\u0abf\u0a82\u0aa4\u0abe \u0aa8\u0ab2\u0abe\u0a96, \u0aac\u0aa7\u0ac1\u0a82 \u0ab8\u0ab0\u0ab8 \u0aa5\u0a88 \u0a9c\u0ab6\u0ac7. \u0ab9\u0ac1\u0a82 \u0a9b\u0ac1\u0a82 \u0aa8\u0ac7 \u0aa4\u0abe\u0ab0\u0abe \u0ab8\u0abe\u0aa5\u0ac7."),
    ("C_male_desc_hinglish", "Ira",
     "a deep male voice, middle-aged Gujarati man, unhurried drawl, warm, casual register, completely unfazed",
     "Thai jase bhai, evu to thatu rhe lya. Bas dil se bolne ka, sab set ho jayega."),
]

for name, speaker, desc, text in LINES:
    print(f"\n=== {name} ===", flush=True)
    prompt = f'<text>{speaker}: <description="{desc}"> {text}<audio>'
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    tg0 = time.time()
    with torch.inference_mode():
        ids = model.generate_audio(**inputs, max_new_tokens=2048,
                                   temperature=0.8, top_k=30, do_sample=True)
    gen_s = time.time() - tg0
    audio_tokens = ids[0].tolist()[inputs.input_ids.shape[1]:]
    codes = model.audio_tokens_to_codes(audio_tokens)
    with torch.inference_mode():
        wav = mimi.decode(codes.to(mimi.device)).audio_values[0, 0]
    path = os.path.join(OUT, f"{name}.wav")
    sf.write(path, wav.float().cpu().numpy(), 24000)
    dur = len(audio_tokens) / 100.0
    print(f"  {len(audio_tokens)} tokens -> {dur:.1f}s audio | gen {gen_s:.1f}s ({dur/gen_s:.2f}x realtime) | saved {path}", flush=True)

peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"\npeak VRAM after all: {peak_gb:.2f} GB", flush=True)
print("SMOKE_TEST_DONE", flush=True)
