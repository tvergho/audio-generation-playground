# For the audio-diffusion-pytorch-trainer model
# Pre-encodes audio files into embeddings using CLAP and a pre-trained DMAE and saves them to disk
# Increases speed for the training process

import sys
import torchaudio
import os
import torch
from tqdm import tqdm
sys.path.insert(0, '/workspace/CLAP/src')

from infer import infer_audio
from audio_data_pytorch import AllTransform
from transformers import AutoModel
from typing import List

alltransformer = AllTransform(random_crop_size=2**20, stereo=True, source_rate=44100, target_rate=48000)
autoencoder = AutoModel.from_pretrained(
    "archinetai/dmae1d-ATC32-v3", trust_remote_code=True
)
autoencoder.requires_grad_(False)
data_dir = "/workspace/data"
out_dir = "/workspace/embeddings"
os.makedirs(out_dir, exist_ok=True)

def fast_scandir(path: str, exts: List[str], recursive: bool = False):
    # Scan files recursively faster than glob
    # From github.com/drscotthawley/aeiou/blob/main/aeiou/core.py
    subfolders, files = [], []

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(path):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in exts:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, exts, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)  # type: ignore

    return subfolders, files

_, files = fast_scandir(data_dir, [".wav"], recursive=True)

# files = files[:100]

existing_files = os.listdir(out_dir)
existing_files.sort(key = lambda x : int(x.split('.')[0]), reverse=True)

if len(existing_files) > 0:
    start = int(existing_files[0].split('.')[0])
else:
    start = 0

for i, filename in tqdm(enumerate(files, start=start), total=len(files)):
    with torch.no_grad():
        waveform, rate = torchaudio.load(filename)
        x = alltransformer(waveform)
        embed = infer_audio([x])
        encoded = autoencoder.encode(x.unsqueeze(dim=0))
        output = { "waveform": encoded.squeeze(), "embedding": embed }
        torch.save(output, f"{out_dir}/{i}.pt")
    
    os.remove(filename)