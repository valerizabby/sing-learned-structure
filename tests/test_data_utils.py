import torch
import pytest

from SingLS.trainer.data_utils import get_chroma, SSM, batch_SSM, topk_sample_one


def test_get_chroma_shape(piano_roll):
    chroma = get_chroma(piano_roll, piano_roll.shape[0])
    assert chroma.shape == (64, 12)


def test_get_chroma_nonnegative(piano_roll):
    chroma = get_chroma(piano_roll, piano_roll.shape[0])
    assert (chroma >= 0).all()


def test_get_chroma_nonzero_for_active_notes(piano_roll):
    chroma = get_chroma(piano_roll, piano_roll.shape[0])
    assert chroma.sum() > 0


def test_ssm_shape(piano_roll):
    ssm = SSM(piano_roll)
    assert ssm.shape == (64, 64)


def test_ssm_symmetric(piano_roll):
    ssm = SSM(piano_roll)
    assert torch.allclose(ssm, ssm.T, atol=1e-5)


def test_ssm_diagonal_near_one():
    # Use a dense roll so no zero rows (zero rows get sim=0, not 1)
    roll = torch.ones(16, 128)
    ssm = SSM(roll)
    assert (ssm.diag() > 0.99).all()


def test_ssm_range(piano_roll):
    ssm = SSM(piano_roll)
    assert ssm.min() >= -0.01
    assert ssm.max() <= 1.01


def test_batch_ssm_single_sample(piano_roll):
    # batch_SSM expects [T, B, 128]
    seq = piano_roll.unsqueeze(1)  # [64, 1, 128]
    result = batch_SSM(seq, batch_size=1)
    assert result.shape == (64, 64)


def test_topk_sample_one_shape():
    logits = torch.randn(1, 1, 128)
    sample = topk_sample_one(logits, k=10, temperature=1.5)
    assert sample.shape == (1, 1, 128)


def test_topk_sample_one_valid_note_range():
    logits = torch.randn(1, 1, 128)
    sample = topk_sample_one(logits, k=10, temperature=1.5)
    active = torch.nonzero(sample[0, 0])
    assert len(active) == 1
    note = active[0, 0].item()
    assert 20 <= note <= 107
