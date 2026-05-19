import torch


def build_ssm_batch(output_len: int, batch_size: int, batched_ssm: torch.Tensor, device) -> torch.Tensor:
    # batched_ssm is tiled [B*ssm_size, ssm_size]; reconstruct [B, T, T]
    T = output_len
    ssm_size = batched_ssm.shape[1]

    SSM = []
    for i in range(batch_size):
        offset = i * ssm_size
        ssm_slice = batched_ssm[offset:offset + T, :T]    # [T, T]
        SSM.append(ssm_slice.unsqueeze(0))                # [1, T, T]

    return torch.cat(SSM, dim=0).to(device)               # [B, T, T]


def set_structure_requires_grad(model, requires_grad: bool) -> None:
    if not hasattr(model, "structure_model") or model.structure_model is None:
        return
    for p in model.structure_model.parameters():
        p.requires_grad = requires_grad


def freeze_structure(model) -> None:
    set_structure_requires_grad(model, False)


def unfreeze_structure(model) -> None:
    set_structure_requires_grad(model, True)