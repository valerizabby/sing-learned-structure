"""
Grid search по весу SSM-лосса (ssm_lambda).

4 прогона × 30 эпох. Каждый прогон:
  - та же архитектура (SING ORIGINAL)
  - тот же датасет (combined_train)
  - только λ отличается

Запуск всех 4 прогонов последовательно:
    python3 -m experiments.lambda_search

Запуск одного конкретного прогона (например λ=10):
    python3 -m experiments.lambda_search --lambda_val 10

После завершения сравнить кривые:
    python3 -m inference.plot_training \\
        --csv data/logs/sing_lambda_1_*_metrics.csv \\
               data/logs/sing_lambda_5_*_metrics.csv \\
               data/logs/sing_lambda_10_*_metrics.csv \\
               data/logs/sing_lambda_20_*_metrics.csv \\
        --labels "λ=1" "λ=5" "λ=10" "λ=20" \\
        --out outputs/lambda_search.png

Потом прогнать метрики:
    python3 -m inference.quick_eval   (подставив нужные чекпоинты)
"""

import argparse
import os
import logging

import torch

from SingLS.config.config import (
    DEVICE, EXP_PATH, EXP_PATH_COMBINED,
    AttentionType, hidden_size, lr, output_size,
)
from SingLS.model.model import MusicGenerator
from SingLS.trainer.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Конфиг эксперимента ─────────────────────────────────────────────────────

NUM_EPOCHS   = 30
ATTENTION    = AttentionType.ORIGINAL        # SING (лучший IOU в ablation)
DATA_PATH    = os.path.join(EXP_PATH_COMBINED, "combined_train.pt")

# λ=1 — текущий baseline (SSM ~2% total)
# λ=5 — умеренный          (SSM ~11%)
# λ=10 — сильный            (SSM ~20%)
# λ=20 — агрессивный        (SSM ~33%)
LAMBDA_GRID = [1, 5, 10, 20]


def run_one(ssm_lambda: float, force: bool = False) -> str:
    """
    Обучает модель с заданным ssm_lambda, возвращает путь к сохранённому чекпоинту.
    Если model_final.pt уже существует — пропускает прогон (если не force=True).
    """
    save_name = f"sing_lambda_{int(ssm_lambda)}"
    save_dir  = os.path.join(EXP_PATH, f"meta_info/lambda_search/{save_name}")
    os.makedirs(save_dir, exist_ok=True)
    final_ckpt = os.path.join(save_dir, "model_final.pt")

    if not force and os.path.exists(final_ckpt):
        logging.info(f"SKIP λ={ssm_lambda}: model_final.pt уже существует → {final_ckpt}")
        return final_ckpt

    logging.info("=" * 60)
    logging.info(f"RUN: {save_name}  (λ={ssm_lambda})")
    logging.info("=" * 60)

    torch.manual_seed(2022)

    data = torch.load(DATA_PATH, weights_only=False)
    logging.info(f"Data loaded: {len(data)} tracks")

    model = MusicGenerator(
        hidden_size=hidden_size,
        output_size=output_size,
        attention_type=ATTENTION,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = ModelTrainer(
        generator=model,
        optimizer=optimizer,
        data=data,
        hidden_size=hidden_size,
        ssm_lambda=ssm_lambda,
    )

    trainer.train_epochs(
        num_epochs=NUM_EPOCHS,
        full_training=True,
        variable_size_batches=True,
        save_name=save_name,
    )

    torch.save(model, final_ckpt)
    logging.info(f"Final model saved → {final_ckpt}")
    return final_ckpt


def main():
    parser = argparse.ArgumentParser(description="Lambda grid search for SSM loss weight")
    parser.add_argument(
        "--lambda_val", type=float, default=None,
        help="Запустить один прогон с заданным λ. Без аргумента — все 4 прогона.",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Перезапустить даже если model_final.pt уже существует.",
    )
    args = parser.parse_args()

    if args.lambda_val is not None:
        run_one(args.lambda_val, force=args.force)
    else:
        results = {}
        for lam in LAMBDA_GRID:
            ckpt = run_one(lam, force=args.force)
            results[lam] = ckpt

        logging.info("\n" + "=" * 60)
        logging.info("LAMBDA SEARCH COMPLETE")
        logging.info("=" * 60)
        for lam, ckpt in results.items():
            logging.info(f"  λ={lam:<4}  →  {ckpt}")
        logging.info("\nДля сравнения кривых:")
        logging.info("  python3 -m inference.plot_training \\")
        logging.info("    --csv data/logs/sing_lambda_*_metrics.csv \\")
        logging.info("    --labels 'λ=1' 'λ=5' 'λ=10' 'λ=20' \\")
        logging.info("    --out outputs/lambda_search.png")


if __name__ == "__main__":
    main()
