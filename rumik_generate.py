"""Stage 3: rumik-oss offline batch renderer (demo clips / benchmarks).

Loads rumik-oss-1 once in 4-bit, renders a batch of texts to WAV, unloads.
Not a live service — generation runs ~0.1x realtime on a 4GB laptop GPU,
so this is for offline clips, not conversation.

Usage:
    python rumik_generate.py --speaker Ira --description "sad, Hindi accent, slow pace" \
        --text "माँ, आज फिर..." --out clip.wav
    python rumik_generate.py --batch batch.jsonl --outdir clips/
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MimiModel

REPO = "rumik-ai/rumik-oss-1"
PROJECT = os.path.dirname(os.path.abspath(__file__))


def load_model():
    print(f"loading {REPO} in 4-bit (bitsandbytes)...", flush=True)
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, trust_remote_code=True, dtype=torch.bfloat16,
        quantization_config=bnb, device_map={"": 0},
    ).eval()
    mimi = MimiModel.from_pretrained(REPO, subfolder="codec").eval().cuda()
    return tok, model, mimi


def synth_one(tok, model, mimi, text, speaker="Ira", description="neutral, steady pace",
              temperature=0.8, top_k=30):
    prompt = f'<text>{speaker}: <description="{description}"> {text}<audio>'
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        ids = model.generate_audio(**inputs, max_new_tokens=2048,
                                   temperature=temperature, top_k=top_k, do_sample=True)
    gen_s = time.time() - t0
    audio_tokens = ids[0].tolist()[inputs.input_ids.shape[1]:]
    codes = model.audio_tokens_to_codes(audio_tokens)
    with torch.inference_mode():
        wav = mimi.decode(codes.to(mimi.device)).audio_values[0, 0]
    dur = len(audio_tokens) / 100.0
    return wav.float().cpu().numpy(), dur, gen_s


def main():
    ap = argparse.ArgumentParser(description="rumik-oss-1 offline batch renderer")
    ap.add_argument("--speaker", default="Ira")
    ap.add_argument("--description", default="neutral, steady pace")
    ap.add_argument("--text", default=None, help="single text to render")
    ap.add_argument("--batch", default=None, help="jsonl with {name,text,speaker,description}")
    ap.add_argument("--out", default=None, help="output wav for --text")
    ap.add_argument("--outdir", default=os.path.join(PROJECT, "assets", "rumik_generated"))
    args = ap.parse_args()

    tok, model, mimi = load_model()
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"loaded | VRAM allocated: {vram:.2f} GB", flush=True)

    jobs = []
    if args.text:
        jobs.append({"name": (args.out or "clip").replace(".wav", "").split(os.sep)[-1],
                     "text": args.text, "speaker": args.speaker,
                     "description": args.description, "out": args.out})
    elif args.batch:
        with open(args.batch, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item.setdefault("speaker", args.speaker)
                item.setdefault("description", args.description)
                item.setdefault("out", os.path.join(args.outdir, f"{item['name']}.wav"))
                jobs.append(item)
    else:
        ap.error("--text or --batch required")

    os.makedirs(args.outdir, exist_ok=True)
    for job in jobs:
        wav, dur, gen_s = synth_one(tok, model, mimi, job["text"],
                                    job["speaker"], job["description"])
        out = job.get("out") or os.path.join(args.outdir, f"{job['name']}.wav")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        sf.write(out, wav, 24000)
        print(f"  {job['name']}: {dur:.1f}s audio in {gen_s:.1f}s "
              f"({dur/gen_s:.2f}x realtime) -> {out}", flush=True)

    torch.cuda.empty_cache()
    print("UNLOADING (gc) — RAM/VRAM released", flush=True)
    del tok, model, mimi


if __name__ == "__main__":
    main()
