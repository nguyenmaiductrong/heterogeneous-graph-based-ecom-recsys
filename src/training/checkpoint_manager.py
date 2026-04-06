import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb

logger = logging.getLogger(__name__)

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"


def _log_ok(msg: str)   -> None: logger.info("%s[OK]%s %s", _GREEN, _RESET, msg)
def _log_err(msg: str)  -> None: logger.error("%s[FAIL]%s %s", _RED, _RESET, msg)
def _log_warn(msg: str) -> None: logger.warning("%s[WARN]%s %s", _YELLOW, _RESET, msg)


class CheckpointManager:
    """
    Quan ly luu/phuc hoi checkpoint BAGNN qua W&B Artifacts, khong can Google Drive.

    Moi chu ky save: ghi local -> upload -> wait() block -> verify API -> xoa local cu.
    File local chi bi xoa sau khi W&B Public API xac nhan artifact da COMMITTED.
    """

    _RUN_ID_FILE = ".wandb_run_id"

    def __init__(
        self,
        project: str,
        entity: str,
        run_name: str = "training-run",
        artifact_name: str = "model-checkpoint",
        save_every_n_epochs: int = 5,
        local_dir: str = "/content/checkpoints",
        verify_timeout_secs: int = 300,
        verify_poll_secs: int = 30,
    ):
        self.project              = project
        self.entity               = entity
        self.run_name             = run_name
        self.artifact_name        = artifact_name
        self.save_every_n_epochs  = save_every_n_epochs
        self.local_dir            = Path(local_dir)
        self.verify_timeout_secs  = verify_timeout_secs
        self.verify_poll_secs     = verify_poll_secs

        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.run: wandb.sdk.wandb_run.Run | None = None
        self._last_logged_artifact: wandb.Artifact | None = None
        self._last_local_ckpt: Path | None = None

    def init_wandb(self, config: dict | None = None) -> wandb.sdk.wandb_run.Run:
        run_id = self._load_run_id()
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            id=run_id,
            resume="allow",
            config=config or {},
        )
        self._save_run_id(self.run.id)
        logger.info("W&B run ready: id=%s", self.run.id)
        return self.run

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None = None,
        device: torch.device | None = None,
    ) -> int:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = self._download_latest_artifact()
        if ckpt_path is None:
            logger.info("No checkpoint found on W&B — starting from epoch 0.")
            return 0

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        logger.info("Resumed epoch %d (loss=%.4f)", ckpt["epoch"], ckpt.get("loss", float("nan")))
        return ckpt["epoch"] + 1

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        scaler: torch.amp.GradScaler | None = None,
        loss: float = float("nan"),
        metrics: dict | None = None,
    ) -> bool:
        """
        Pipeline: luu local -> upload -> verify cloud -> cleanup.

        Tra ve True neu cloud xac nhan thanh cong.
        Tra ve False neu verify that bai — file local duoc giu lai
        """
        if (epoch + 1) % self.save_every_n_epochs != 0:
            return True

        if self.run is None:
            raise RuntimeError("Call init_wandb() first.")

        ckpt_path       = self._save_local(model, optimizer, scaler, epoch, loss, metrics)
        logged_artifact = self._upload_artifact(ckpt_path, epoch, loss, metrics)

        self._last_logged_artifact = logged_artifact
        self._last_local_ckpt      = ckpt_path

        verified = self.verify_checkpoint_on_cloud(
            logged_artifact=logged_artifact,
            local_path=ckpt_path,
            epoch=epoch,
        )

        if verified:
            self._cleanup_old_checkpoints(keep=ckpt_path)
        else:
            _log_err(
                f"Epoch {epoch}: not confirmed on cloud. "
                f"File retained at: {ckpt_path}. DO NOT CLOSE COLAB."
            )

        return verified

    def verify_checkpoint_on_cloud(
        self,
        logged_artifact: wandb.Artifact | None = None,
        local_path: Path | None = None,
        epoch: int | None = None,
        expected_version: str = "latest",
    ) -> bool:
        logged_artifact = logged_artifact or self._last_logged_artifact
        local_path = local_path or self._last_local_ckpt

        label = f"epoch {epoch}" if epoch is not None else expected_version
        artifact_ref = f"{self.entity}/{self.project}/{self.artifact_name}:{expected_version}"

        if logged_artifact is not None:
            try:
                logged_artifact.wait()
            except Exception as exc:
                _log_warn(f"artifact.wait() failed ({exc}), proceeding to API check.")

        local_size = local_path.stat().st_size if (local_path and local_path.exists()) else None
        deadline = time.monotonic() + self.verify_timeout_secs
        attempt = 0

        while time.monotonic() < deadline:
            attempt += 1
            try:
                cloud_artifact = wandb.Api(timeout=60).artifact(artifact_ref, type="model")
                state = getattr(cloud_artifact, "state", "UNKNOWN").upper()

                if state not in ("COMMITTED", "READY"):
                    _log_warn(f"[{attempt}] state={state}, retry in {self.verify_poll_secs}s...")
                    time.sleep(self.verify_poll_secs)
                    continue

                cloud_size = self._cloud_size(cloud_artifact)
                if not self._size_ok(local_size, cloud_size, label, attempt):
                    time.sleep(self.verify_poll_secs)
                    continue

                version = getattr(cloud_artifact, "version", expected_version)
                _log_ok(
                    f"Checkpoint {label} -> v{version} confirmed "
                    f"(state={state}, {cloud_size / 1e6:.1f} MB). Safe to close Colab."
                )
                return True

            except wandb.errors.CommError as exc:
                _log_warn(f"[{attempt}] CommError: {exc}, retry in {self.verify_poll_secs}s...")
                time.sleep(self.verify_poll_secs)
            except Exception as exc:
                _log_warn(f"[{attempt}] {type(exc).__name__}: {exc}, retry in {self.verify_poll_secs}s...")
                time.sleep(self.verify_poll_secs)

        _log_err(
            f"Checkpoint {label} NOT confirmed after {self.verify_timeout_secs}s "
            f"({attempt} attempts). Ref: {artifact_ref}. "
            f"DO NOT CLOSE COLAB — try again in a few minutes."
        )
        return False

    def _save_local(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        epoch: int,
        loss: float,
        metrics: dict | None,
    ) -> Path:
        path = self.local_dir / f"epoch_{epoch:03d}.pt"
        state: dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "metrics": metrics or {},
        }
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        torch.save(state, path)
        logger.info("Saved local: %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path

    def _upload_artifact(
        self,
        ckpt_path: Path,
        epoch: int,
        loss: float,
        metrics: dict | None,
    ) -> wandb.Artifact:
        artifact = wandb.Artifact(
            name=self.artifact_name,
            type="model",
            metadata={"epoch": epoch, "loss": loss, **(metrics or {})},
        )
        artifact.add_file(str(ckpt_path))
        logged = self.run.log_artifact(artifact, aliases=["latest", f"epoch-{epoch:03d}"])
        logger.info("Artifact enqueued: %s epoch-%03d", self.artifact_name, epoch)
        return logged

    def _cloud_size(self, cloud_artifact: wandb.Artifact) -> int:
        size = getattr(cloud_artifact, "size", None)
        if isinstance(size, (int, float)) and size > 0:
            return int(size)
        try:
            return sum(f.size for f in cloud_artifact.files())
        except Exception:
            return 0

    def _size_ok(self, local: int | None, cloud: int, label: str, attempt: int) -> bool:
        if local is None:
            _log_warn("Local file not found, skipping size check.")
            return True
        if cloud == 0:
            _log_warn(f"[{attempt}] Cloud size=0, upload may be incomplete.")
            return False
        diff = abs(cloud - local) / max(local, 1)
        if diff > 0.01:
            _log_warn(
                f"[{attempt}] Size mismatch ({label}): "
                f"local={local / 1e6:.2f} MB, cloud={cloud / 1e6:.2f} MB ({diff * 100:.1f}%)."
            )
            return False
        logger.info("Size OK: %.2f MB (diff %.2f%%)", cloud / 1e6, diff * 100)
        return True

    def _download_latest_artifact(self) -> Path | None:
        if self.run is None:
            raise RuntimeError("Call init_wandb() first.")

        ref = f"{self.entity}/{self.project}/{self.artifact_name}:latest"
        try:
            dl_dir = Path(self.run.use_artifact(ref, type="model").download(root=str(self.local_dir)))
        except wandb.errors.CommError:
            logger.info("Artifact %s not found — fresh start.", ref)
            return None

        pt_files = list(dl_dir.glob("*.pt")) + list(dl_dir.glob("*.pth"))
        if not pt_files:
            logger.warning("No .pt file in downloaded artifact dir: %s", dl_dir)
            return None
        return max(pt_files, key=lambda p: p.stat().st_mtime)

    def _cleanup_old_checkpoints(self, keep: Path) -> None:
        for pt in list(self.local_dir.glob("*.pt")) + list(self.local_dir.glob("*.pth")):
            if pt.resolve() != keep.resolve():
                pt.unlink()
                logger.info("Removed old checkpoint: %s", pt)

    def _load_run_id(self) -> str | None:
        id_file = self.local_dir / self._RUN_ID_FILE
        if not id_file.exists():
            return None
        run_id = id_file.read_text().strip()
        logger.info("Resuming W&B run: %s", run_id)
        return run_id

    def _save_run_id(self, run_id: str) -> None:
        (self.local_dir / self._RUN_ID_FILE).write_text(run_id)