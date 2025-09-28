## Prepares WDS tars for training 
from pathlib import Path
import shutil

INPUT_DIR = Path("/fast/ntomasz/data/gen_animals_v2/wds_cd_v2/0/3")
OUT_DIR = Path("/tmp/wds") # /tmp not necessary
NUM_TARS = 400

OUT_DIR.mkdir(exist_ok=True)

tar_files = sorted(INPUT_DIR.rglob("*.tar"))[:NUM_TARS]

for i, tar_path in enumerate(tar_files):
    shutil.copy(tar_path, OUT_DIR / f"{i:06d}.tar")
    print(f"{i+1}/{len(tar_files)} {tar_path.name} -> {i:06d}.tar")
