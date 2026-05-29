import wandb
from pathlib import Path

entity = "nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng"
project = "bpatmp-recsys"
artifact_name = "bpatmp-final-l1"  # đổi thành bpatmp-final-l4 nếu cần

api = wandb.Api()

artifact = api.artifact(
    f"{entity}/{project}/{artifact_name}:epoch-003",
    type="model"
)

download_dir = artifact.download(root="checkpoints/downloaded")
print("Downloaded to:", download_dir)

pt_files = list(Path(download_dir).glob("*.pt")) + list(Path(download_dir).glob("*.pth"))
print("Model files:")
for p in pt_files:
    print(p)