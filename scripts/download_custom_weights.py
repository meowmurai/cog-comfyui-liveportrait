
import time
import os
from huggingface_hub import snapshot_download

COMFYUI_MODEL_DIR = "ComfyUI/models"

def main():
    # Downloading latest weights
    base_bath = os.path.join(COMFYUI_MODEL_DIR, "liveportrait")

    start_time = time.time()
    print(f"Downloading liveportrait models to: {base_bath}")
    # Weights for human modes
    saved_path_human = snapshot_download(
        repo_id="Kijai/LivePortrait_safetensors",
        ignore_patterns=["*animal*"],
        local_dir=base_bath,
        local_dir_use_symlinks=False,
    )
    elapsed_time = time.time() - start_time
    print(
            f"✅ liveportrait human checkpoints downloaded to {saved_path_human} in {elapsed_time:.2f}s"
    )
    # Weights for animal modes

    start_time = time.time()
    saved_path_animal = snapshot_download(
        repo_id="phuc307/liveportrait-safetensors",
        allow_patterns=["*animal*"],
        local_dir=base_bath,
        local_dir_use_symlinks=False,
    )
    elapsed_time = time.time() - start_time
    print(
        f"✅ liveportrait animal checkpoints downloaded to {saved_path_animal}/animal in {elapsed_time:.2f}s"
    )
if __name__ == "__main__":
    main()
