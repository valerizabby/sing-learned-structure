import sys
import time
import os
import logging
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
    filemode="w",
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logging.getLogger().addHandler(console)


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

def custom_loss(output, target):
    criterion = nn.BCEWithLogitsLoss()

    bce_loss = criterion(output, target.float())
    batch_size = output.size(1)
    ssm_err = 0

    for i in range(batch_size):
        SSM1 = SSM(output[:, i, :])
        SSM2 = SSM(target[:, i, :])
        diff = (SSM1 - SSM2) ** 2
        ssm_loss = torch.sum(diff) / (SSM2.size(0) ** 2)
        ssm_err += ssm_loss

        if i == 0:
            logging.info(f"[Loss Debug] Sample {i}")
            logging.info(f"SSM1 mean: {SSM1.mean():.4f}, SSM2 mean: {SSM2.mean():.4f}")
            logging.info(f"SSM loss (sample 0): {ssm_loss:.6f}")

    total_loss = bce_loss + ssm_err
    logging.info(
        f"[Loss Debug] BCE loss: {bce_loss.item():.6f}, SSM error: {ssm_err.item():.6f}, Total: {total_loss.item():.6f}"
    )
    return total_loss


class ModelTrainer:
    def __init__(self, generator, optimizer, data, hidden_size=128, batch_size=50):
        self.generator = generator
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.data = data
        self.data_length = data[0][0].shape[0]

        # alpha scheduling
        # self.alpha_max = 0.05
        # self.alpha_warmup_epochs = 10
        # self.alpha_history = []

        # struct loss weight
        self.beta_struct = 0.03

        logging.info("Starting training with config:")
        logging.info(f"  Attention type: {self.generator.attention_type}")
        logging.info(f"  Hidden size: {self.hidden_size}")
        logging.info(f"  Data shape: {len(self.data)}")
        logging.info(f"  Random seed: 2022")

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

        logging.info("Starting training loop")
        logging.info(f"  Epochs: {num_epochs}")
        logging.info(f"  Full training: {full_training}")
        logging.info(f"  Variable size batches: {variable_size_batches}")
        logging.info(f"  Model attention: {self.generator.attention_type}")

        try:
            # стабилизация локальной генерации
            freeze_structure(self.generator)

            for epoch in tqdm(range(num_epochs), desc="Epochs", dynamic_ncols=True):
                # self._update_alpha(epoch)
                # alpha_val = self._get_alpha_value()
                # if alpha_val is not None:
                #     self.alpha_history.append(alpha_val)
                #     logging.info(f"[Alpha] epoch={epoch + 1} alpha={alpha_val:.6f}")

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

                cum_loss = 0.0

                for batch in tqdm(batches, desc=f"Epoch {epoch + 1}", leave=False, dynamic_ncols=True):
                    if full_training:
                        loss = self.train(batch)
                    else:
                        loss = self.train(batch[:, :105, :])
                    cum_loss += loss
                    del batch, loss

                epoch_loss = cum_loss / len(batches)
                losslist.append(epoch_loss)

                elapsed = time.time() - start_time
                msg = f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.5f} | Time: {elapsed:.2f}s"
                tqdm.write(msg)
                logging.info(msg)

                save_dir = os.path.join("models", "checkpoints")
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{save_name}-epoch-{epoch + 1:02d}-loss-{epoch_loss:.5f}.pt"
                path = os.path.join(save_dir, filename)

                torch.save(self.generator, path)
                tqdm.write(f"Model checkpoint saved to {path}")
                logging.info(f"Model checkpoint saved to {path}")

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

        single_loss = custom_loss(output_full, target_full)

        structure_model = self._get_structure_model()
        # вызываем build struct в 25% случаев
        if structure_model is not None:
            MAX_T = 256
            T_struct = min(min_len, MAX_T)

            # случайное окно по времени
            if min_len > T_struct:
                start = torch.randint(0, min_len - T_struct + 1, (1,)).item()
            else:
                start = 0
            ssm_batch = build_ssm_batch(
                output_len=T_struct,
                batch_size=batch_size,
                batched_ssm=self_sim,
                device=output_full.device,
            )  # [B, T, T]
            logging.info("[Structure Model] Build struct.")
            with torch.no_grad():
                struct = structure_model(ssm_batch)
            struct = struct[start:start + T_struct]

            target_crop = target_full[start:start + T_struct]
            logging.info("[Structure Model] Compute loss.")
            s_loss = structure_loss(struct, target_crop)
            single_loss = single_loss + self.beta_struct * s_loss
            logging.info(f"[StructLoss] struct_loss={s_loss.item():.6f}, beta={self.beta_struct:.4f}")

        single_loss.backward()
        self.optimizer.step()

        loss += float(single_loss.detach().to(DEVICE))

        del next_element, self_sim, sequence, generated, single_loss
        return loss

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