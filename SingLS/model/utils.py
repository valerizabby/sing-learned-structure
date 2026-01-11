import torch

def build_ssm_batch(output_len: int, batch_size: int, batched_ssm: torch.Tensor, device) -> torch.Tensor:
    # batched_ssm у тебя плиткой; восстанавливаем [B, T, T]
    T = output_len
    ssm_size = batched_ssm.shape[1]

    SSM = []
    for i in range(batch_size):
        offset = i * ssm_size
        ssm_slice = batched_ssm[offset:offset + T, :T]    # [T, T]
        SSM.append(ssm_slice.unsqueeze(0))                # [1, T, T]
        if not torch.isfinite(ssm_slice).all():
            print(f"SSM slice for sample {i} has NaN/Inf")

    return torch.cat(SSM, dim=0).to(device)               # [B, T, T]

def freeze_structure(model):
    if not hasattr(model, "structure_model"):
        return
    if model.structure_model is None:
        return
    for p in model.structure_model.parameters():
        p.requires_grad = False


def unfreeze_structure(model):
    if not hasattr(model, "structure_model"):
        return
    if model.structure_model is None:
        return
    for p in model.structure_model.parameters():
        p.requires_grad = True