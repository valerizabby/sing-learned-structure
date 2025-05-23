import os
import torch
import logging
import matplotlib.pyplot as plt

from SingLS.config.config import DEVICE, EXP_PATH, AttentionType, num_epochs, hidden_size, lr, output_size
from SingLS.model.model import MusicGenerator
from SingLS.trainer.train import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == '__main__':
    data_path = os.path.join(EXP_PATH, "mar-1-variable_bin_bounds_train.csv")
    model_save_path = os.path.join(EXP_PATH, "trained_alpha_lsa", f"model_{num_epochs}_epochs.txt")
    plot_save_path = os.path.join(EXP_PATH, "plots", "train_loss.png")

    logging.info(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    logging.info("Data loaded successfully.")

    torch.manual_seed(2022)

    logging.info("Initializing model...")

    generator = MusicGenerator(
        hidden_size=hidden_size,
        output_size=output_size,
        attention_type=AttentionType.LSA
    ).to(DEVICE)

    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

    logging.info(f"Starting training for {num_epochs} epochs...")

    trainer = ModelTrainer(generator, optimizer, data, hidden_size)
    losslist, piclist = trainer.train_epochs(
        num_epochs=num_epochs,
        full_training=True,
        variable_size_batches=True
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(generator, model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.scatter(range(len(losslist)), losslist)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(plot_save_path)
    plt.close()
    logging.info(f"Training loss plot saved to {plot_save_path}")