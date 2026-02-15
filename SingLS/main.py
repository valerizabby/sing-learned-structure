import os
import torch
import logging
import matplotlib.pyplot as plt

from SingLS.config.config import DEVICE, EXP_PATH, AttentionType, num_epochs, hidden_size, lr, output_size, \
    EXP_PATH_LMD, EXP_PATH_COMBINED, struct_lr
from SingLS.model.hierarchical_generator import HierarchicalGenerator
from SingLS.model.model import MusicGenerator
from SingLS.model.structure_transformer import StructureTransformer, StructureModel
from SingLS.trainer.train import ModelTrainer
from enum import Enum

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Model(Enum):
    LSA_SB = (AttentionType.LSA_SB, "lsa_sb") # +
    LSA = (AttentionType.LSA, "lsa")
    ORIGINAL = (AttentionType.ORIGINAL, "original")
    NONE = (AttentionType.NONE, "none")

    TRANSFORMER_LSA_SB = (AttentionType.LSA_SB, "transformer_lsa_sb") # +
    TRANSFORMER_LSA = (AttentionType.LSA, "transformer_lsa")
    TRANSFORMER_ORIGINAL = (AttentionType.ORIGINAL, "transformer_original")
    TRANSFORMER = (AttentionType.NONE, "transformer")


if __name__ == '__main__':
    CURRENT_MODEL = Model.LSA

    data_path = os.path.join(EXP_PATH_COMBINED, "combined_train.pt")
    model_save_path = os.path.join(EXP_PATH, f"meta_info/trained_{CURRENT_MODEL.value[1]}_combined", f"model_{num_epochs}_epochs.txt")
    plot_save_path = os.path.join(EXP_PATH, "meta_info/plots", f"train_loss_{CURRENT_MODEL.value[1]}_combined.png")

    logging.info(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    logging.info("Data loaded successfully.")

    torch.manual_seed(2022)

    logging.info("Initializing model...")

    model = MusicGenerator(
        hidden_size=hidden_size,
        output_size=output_size,
        attention_type=CURRENT_MODEL.value[0]
    )

    # generator = MusicGenerator(
    #     hidden_size=hidden_size,
    #     output_size=output_size,
    #     attention_type=CURRENT_MODEL.value[0]
    # )
    #
    # structure_transformer = StructureTransformer(
    #     d_model=hidden_size,
    #     nhead=4,
    #     num_layers=2
    # )
    #
    # structure_model = StructureModel(
    #     transformer=structure_transformer,
    #     proj=torch.nn.Linear(hidden_size, hidden_size)
    # )
    #
    # model = HierarchicalGenerator(
    #     generator=generator,
    #     structure_model=structure_model
    # ).to(DEVICE)
    #
    # gen_params = model.generator.parameters()
    #
    # struct_params = (
    #     model.structure_model.parameters()
    #     if model.structure_model is not None
    #     else []
    # )
    # optimizer = torch.optim.Adam(
    #     [
    #         {"params": gen_params, "lr": lr},
    #         {"params": struct_params, "lr": struct_lr},
    #     ]
    # )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logging.info(f"Starting training for {num_epochs} epochs...")

    trainer = ModelTrainer(model, optimizer, data, hidden_size)
    losslist, piclist = trainer.train_epochs(
        num_epochs=num_epochs,
        full_training=True,
        variable_size_batches=True
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    torch.save(model, model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.scatter(range(len(losslist)), losslist)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(plot_save_path)
    plt.close()
    logging.info(f"Training loss plot saved to {plot_save_path}")

    if hasattr(trainer, "alpha_history") and len(trainer.alpha_history) > 0:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(trainer.alpha_history)
        plt.xlabel("Epoch")
        plt.ylabel("alpha")
        plt.title("Structure contribution (alpha)")
        plt.grid(True)

        alpha_plot_path = os.path.join(EXP_PATH, "meta_info/plots", "alpha_curve.png")
        plt.savefig(alpha_plot_path)
        plt.close()

        logging.info(f"Alpha curve saved to {alpha_plot_path}")