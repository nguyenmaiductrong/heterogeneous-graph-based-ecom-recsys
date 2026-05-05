from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nguyenmaiductrong/rees46-bpatmp-temporal",
    repo_type="dataset",
    local_dir="data",
    local_dir_use_symlinks=False,
)
