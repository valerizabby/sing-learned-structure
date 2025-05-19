import torch


def original_attention(output, prev_sequence, batched_ssm, attention_layer, sparsemax, device):
    sequence_length, batch_size, _ = output.size()
    avg_output = output[-1, :, :].unsqueeze(1)  # [batch, 1, 128]

    beat_num = prev_sequence.shape[0]
    ssm_size = batched_ssm.shape[1]
    inds_across_pieces = range(beat_num, batched_ssm.shape[0], ssm_size)

    ssm_slice = batched_ssm[inds_across_pieces, :beat_num]  # [batch, beat_num]
    weights = sparsemax(ssm_slice)  # [batch, beat_num]

    weighted = (prev_sequence.permute(2, 1, 0) * weights).T  # [beat_num, batch, 128]
    weight_vec = torch.sum(weighted, dim=0).unsqueeze(1).to(device)  # [batch, 1, 128]

    pt2 = torch.cat((weight_vec, avg_output), dim=1)  # [batch, 2, 128]

    # attention применяется по последнему измерению: [batch, 2, 128] → [batch, 1, 128]
    attentioned = attention_layer(pt2.permute(0, 2, 1))  # → [batch, 128, 1]
    attentioned = attentioned.permute(2, 0, 1)  # → [1, batch, 128]

    return attentioned.double()