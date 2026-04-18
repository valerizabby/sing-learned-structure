import sys
import time
import os
import csv
import json
import logging
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn

from SingLS.config.config import EXP_PATH, DEVICE
from SingLS.model.utils import freeze_structure, unfreeze_structure
from SingLS.trainer.data_utils import (
    batch_SSM,
    topk_batch_sample,
    make_variable_size_batches,
    make_batches,
    SSM,
)
from SingLS.model.utils import build_ssm_batch


LOG_DIR = os.path.join(EXP_PATH, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "training.log"),
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logging.getLogger().addHandler(console)

_CSV_COLUMNS = [
    "epoch", "total_loss", "bce_loss", "ssm_loss", "struct_loss",
    "train_time_s", "n_batches",
]


def structure_loss(struct, target, eps=1e-8):
    """
    struct, target: [T, B, H]
    """
    target = target.float()

    struct_feat = struct.mean(dim=0)   # [B, H]
    target_feat = target.mean(dim=0)   # [B, H]

    struct_feat = struct_feat / (struct_feat.norm(dim=1, keepdim=True) + eps)
    target_feat = target_feat / (target_feat.norm(dim=1, keepdim=True) + eps)

    return 1.0 - (struct_feat * target_feat).sum(dim=1).mean()

def custom_loss(output, target, ssm_lambda=1.0):
    criterion = nn.BCEWithLogitsLoss()

    bce_loss = criterion(output, target.float())
    batch_size = output.size(1)
    ssm_err = 0

    for i in range(batch_size):
        # sigmoid maps logits → [0,1], same domain as binary target.
        # Gradient flows through sigmoid back to model weights.
        # Without this, SSM(logits) can be negative while SSM(binary) ∈ [0,1]
        # — incompatible domains make the SSM loss meaningless.
        output_prob = torch.sigmoid(output[:, i, :])
        SSM1 = SSM(output_prob)
        SSM2 = SSM(target[:, i, :])
        diff = (SSM1 - SSM2) ** 2
        ssm_loss = torch.sum(diff) / (SSM2.size(0) ** 2)
        ssm_err += ssm_loss

        if i == 0:
            logging.debug(
                f"[Loss] sample0 SSM1_mean={SSM1.mean():.4f} SSM2_mean={SSM2.mean():.4f} "
                f"ssm_loss={ssm_loss:.6f}"
            )

    total_loss = bce_loss + ssm_lambda * ssm_err
    logging.debug(
        f"[Loss] bce={bce_loss.item():.6f} ssm={ssm_err.item():.6f} "
        f"lambda={ssm_lambda} total={total_loss.item():.6f}"
    )
    return total_loss, bce_loss.item(), ssm_err.detach().item()


class ModelTrainer:
    def __init__(self, generator, optimizer, data, hidden_size=128, batch_size=50, ssm_lambda=1.0):
        self.generator = generator
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.data = data
        self.data_length = data[0][0].shape[0]
        self.ssm_lambda = ssm_lambda

        # struct loss weight
        self.beta_struct = 0.03

        # будет установлен в train_epochs
        self._csv_path = None   # type: Optional[str]
        self._json_path = None  # type: Optional[str]
        self._run_meta: dict = {}

        logging.info("Starting training with config:")
        logging.info(f"  Attention type: {self.generator.attention_type}")
        logging.info(f"  Hidden size: {self.hidden_size}")
        logging.info(f"  Data shape: {len(self.data)}")
        logging.info(f"  SSM lambda: {self.ssm_lambda}")
        logging.info(f"  Random seed: 2022")

    # ── Structured logging helpers ────────────────────────────────────────────

    def _init_run_logs(self, save_name, num_epochs, extra_meta=None):
        """Создаёт CSV и JSON для текущего прогона обучения."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{save_name}_{ts}"

        self._csv_path = os.path.join(LOG_DIR, f"{run_id}_metrics.csv")
        self._json_path = os.path.join(LOG_DIR, f"{run_id}_config.json")

        self._run_meta = {
            "run_id": run_id,
            "save_name": save_name,
            "started_at": ts,
            "attention_type": str(self.generator.attention_type),
            "hidden_size": self.hidden_size,
            "beta_struct": self.beta_struct,
            "ssm_lambda": self.ssm_lambda,
            "num_epochs": num_epochs,
            "data_size": len(self.data),
            **(extra_meta or {}),
            "epochs": [],
        }

        with open(self._csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_COLUMNS).writeheader()

        with open(self._json_path, "w") as f:
            json.dump(self._run_meta, f, indent=2)

        logging.info(f"[Metrics] CSV  → {self._csv_path}")
        logging.info(f"[Metrics] JSON → {self._json_path}")

    def _log_epoch(self, row: dict):
        """Дописывает строку в CSV и обновляет epochs-список в JSON."""
        if self._csv_path is None:
            return

        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writerow(row)

        self._run_meta["epochs"].append(row)
        self._run_meta["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self._json_path, "w") as f:
            json.dump(self._run_meta, f, indent=2)

    def _get_structure_model(self):
        # HierarchicalGenerator: .structure_model
        sm = getattr(self.generator, "structure_model", None)
        if sm is not None:
            return sm
        return None

    def _set_alpha_value(self, alpha_value: float):
        """
        Robust setter:
        - if alpha is float => assign
        - if alpha is scalar tensor/Parameter => fill_
        """
        if not hasattr(self.generator, "alpha"):
            return

        a = getattr(self.generator, "alpha")
        # alpha can be a property (read-only) in some versions — тогда просто пропускаем
        try:
            if isinstance(a, (float, int)):
                setattr(self.generator, "alpha", float(alpha_value))
                return
            if torch.is_tensor(a):
                # Parameter / Tensor
                if a.numel() == 1:
                    a.data.fill_(float(alpha_value))
                    return
        except Exception:
            return

    def _get_alpha_value(self):
        if not hasattr(self.generator, "alpha"):
            return None
        a = getattr(self.generator, "alpha")
        if isinstance(a, (float, int)):
            return float(a)
        if torch.is_tensor(a) and a.numel() == 1:
            return float(a.detach().cpu().item())
        return None

    def _update_alpha(self, epoch: int):
        """
        Linear warmup: 0 → alpha_max за alpha_warmup_epochs
        epoch: 0..num_epochs-1
        """
        if not hasattr(self.generator, "alpha"):
            return
        alpha_value = min(self.alpha_max, self.alpha_max * float(epoch) / float(self.alpha_warmup_epochs))
        self._set_alpha_value(alpha_value)

    def train_epochs(self, num_epochs=50, full_training=False, variable_size_batches=False, save_name="model"):
        losslist = []
        piclist = []

        self._init_run_logs(save_name, num_epochs, extra_meta={
            "full_training": full_training,
            "variable_size_batches": variable_size_batches,
        })

        logging.info("Starting training loop")
        logging.info(f"  Epochs: {num_epochs}")
        logging.info(f"  Full training: {full_training}")
        logging.info(f"  Variable size batches: {variable_size_batches}")
        logging.info(f"  Model attention: {self.generator.attention_type}")

        try:
            freeze_structure(self.generator)

            for epoch in tqdm(range(num_epochs), desc="Epochs", dynamic_ncols=True):
                if epoch == 5:
                    unfreeze_structure(self.generator)

                self.generator.train()
                start_time = time.time()

                if variable_size_batches:
                    batches = make_variable_size_batches(self.data, 1)
                elif full_training:
                    batches = make_batches(self.data, self.batch_size, self.data_length)
                else:
                    batches = make_batches(self.data[:100], self.batch_size, self.data_length)

                cum_total = 0.0
                cum_bce   = 0.0
                cum_ssm   = 0.0
                cum_struct = 0.0

                for batch in tqdm(batches, desc=f"Epoch {epoch + 1}", leave=False, dynamic_ncols=True):
                    if full_training:
                        total, bce, ssm, struct = self.train(batch)
                    else:
                        total, bce, ssm, struct = self.train(batch[:, :105, :])
                    cum_total  += total
                    cum_bce    += bce
                    cum_ssm    += ssm
                    cum_struct += struct
                    del batch

                n = len(batches)
                epoch_total  = cum_total  / n
                epoch_bce    = cum_bce    / n
                epoch_ssm    = cum_ssm    / n
                epoch_struct = cum_struct / n
                elapsed = time.time() - start_time

                losslist.append(epoch_total)

                msg = (
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"loss={epoch_total:.5f} bce={epoch_bce:.5f} "
                    f"ssm={epoch_ssm:.5f} struct={epoch_struct:.5f} | "
                    f"time={elapsed:.1f}s"
                )
                tqdm.write(msg)
                logging.info(msg)

                row = {
                    "epoch":       epoch + 1,
                    "total_loss":  round(epoch_total,  6),
                    "bce_loss":    round(epoch_bce,    6),
                    "ssm_loss":    round(epoch_ssm,    6),
                    "struct_loss": round(epoch_struct, 6),
                    "train_time_s": round(elapsed,     2),
                    "n_batches":   n,
                }
                self._log_epoch(row)

                save_dir = os.path.join("models", "checkpoints")
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{save_name}-epoch-{epoch + 1:02d}-loss-{epoch_total:.5f}.pt"
                path = os.path.join(save_dir, filename)
                torch.save(self.generator, path)
                logging.info(f"Checkpoint → {path}")

                snap = self.generate_n_examples(n=1, length=95, starter_notes=10)
                piclist.append(snap)

        except Exception:
            logging.exception("Training interrupted due to an error:")

        return losslist, piclist

    def train(self, batch, starter_notes=5):
        batch_size = batch.shape[0]
        self_sim = batch_SSM(batch.transpose(0, 1), batch_size)  # batched SSM
        sequence = batch[:, 0:starter_notes, :].transpose(0, 1)
        generated = batch[:, 0:starter_notes, :].transpose(0, 1)

        self.generator.init_hidden(batch_size)
        self.optimizer.zero_grad()

        loss = 0.0
        next_element = sequence.to(DEVICE)

        # autoregressive generation
        for i in range(0, batch.shape[1] - starter_notes):
            val = torch.rand(1)

            output, _ = self.generator.forward(next_element, batch_size, sequence, self_sim)

            if val > 0.8:
                next_element = batch[:, i + 1, :].unsqueeze(0)
            else:
                next_element = topk_batch_sample(output, 50)

            sequence = torch.vstack((sequence, next_element.to(DEVICE)))
            generated = torch.vstack((generated, output.to(DEVICE)))

        # loss targets
        output_full = generated[starter_notes:, :, :]
        target_full = batch.transpose(0, 1)[starter_notes:, :, :]

        min_len = min(output_full.shape[0], target_full.shape[0])
        output_full = output_full[:min_len]
        target_full = target_full[:min_len]

        single_loss, bce_val, ssm_val = custom_loss(output_full, target_full, ssm_lambda=self.ssm_lambda)
        struct_val = 0.0

        structure_model = self._get_structure_model()
        if structure_model is not None:
            MAX_T = 256
            T_struct = min(min_len, MAX_T)

            if min_len > T_struct:
                start = torch.randint(0, min_len - T_struct + 1, (1,)).item()
            else:
                start = 0
            ssm_batch = build_ssm_batch(
                output_len=T_struct,
                batch_size=batch_size,
                batched_ssm=self_sim,
                device=output_full.device,
            )
            with torch.no_grad():
                struct = structure_model(ssm_batch)
            struct = struct[start:start + T_struct]

            target_crop = target_full[start:start + T_struct]
            s_loss = structure_loss(struct, target_crop)
            single_loss = single_loss + self.beta_struct * s_loss
            struct_val = float(s_loss.detach())
            logging.debug(f"[StructLoss] struct={struct_val:.6f} beta={self.beta_struct}")

        single_loss.backward()
        self.optimizer.step()

        total_val = float(single_loss.detach())
        loss += total_val

        del next_element, self_sim, sequence, generated, single_loss
        return total_val, bce_val, ssm_val, struct_val

    def generate_n_pieces(self, initial_vectors, n_pieces, length, batched_ssm):
        self.generator.eval()
        self.generator.set_random_hidden(n_pieces)

        sequence = initial_vectors.transpose(0, 1)
        next_element = sequence.to(DEVICE)

        max_notes = batched_ssm.shape[0] - sequence.shape[0]

        for _ in range(min(length, max_notes)):
            with torch.no_grad():
                output, _ = self.generator.forward(next_element.float(), n_pieces, sequence, batched_ssm)
                next_element = topk_batch_sample(output, 50)
            sequence = torch.vstack((sequence, next_element.to(DEVICE)))

        return sequence

    def generate_n_examples(self, n=1, length=390, starter_notes=10, piece_inds=[0], random_source_pieces=False):
        pieces = torch.vstack([self.data[i][0].unsqueeze(0) for i in piece_inds])
        first_vecs = pieces[:, 0:starter_notes, :]
        batched_ssms = batch_SSM(pieces.transpose(0, 1), n)

        new_gen = self.generate_n_pieces(first_vecs, n, length, batched_ssms)

        del pieces, first_vecs, batched_ssms
        return new_gen