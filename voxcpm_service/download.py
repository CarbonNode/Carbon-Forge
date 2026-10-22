"""Download a pinned, public model snapshot; this does not load it or use the GPU."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
destination = os.environ.get("VOXCPM_MODEL_DIR", "/models/voxcpm2")
snapshot_download("openbmb/VoxCPM2", revision=REVISION, local_dir=destination,
                  ignore_patterns=["*.md", "*.png", "*.jpg", "*.mp4"], max_workers=2)
Path(destination, ".ready").write_text(REVISION)
