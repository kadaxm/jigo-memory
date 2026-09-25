from huggingface_hub import snapshot_download

p = snapshot_download("rumik-ai/rumik-oss-1")
print("DOWNLOAD_COMPLETE:", p)
