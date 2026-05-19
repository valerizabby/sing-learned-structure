import torch
import pytest

from Seg2SSM.affinity_ssm import AffinitySSM


@pytest.fixture
def builder():
    return AffinitySSM.fixed()


@pytest.fixture
def simple_plan():
    return [(1, 8), (2, 8)]  # verse:8, chorus:8


def test_build_shape(builder, simple_plan):
    ssm = builder.build(simple_plan, ssm_size=32)
    assert ssm.shape == (32, 32)


def test_build_symmetric(builder, simple_plan):
    ssm = builder.build(simple_plan, ssm_size=32)
    assert torch.allclose(ssm, ssm.T, atol=1e-4)


def test_build_range(builder, simple_plan):
    ssm = builder.build(simple_plan, ssm_size=32)
    assert ssm.min() >= 0.0
    assert ssm.max() <= 1.0


def test_build_uniform_labels_high_mean():
    """All-verse plan → high mean similarity (same label throughout)."""
    b = AffinitySSM.fixed(noise_std=0.0, rescale=False)
    ssm = b.build([(1, 32)], ssm_size=32)
    # All bars are verse, so the full SSM should reflect high verse-verse affinity
    assert ssm.mean().item() > 0.7


def test_build_cross_block_lower_than_diagonal():
    """verse vs bridge cross-block < verse vs verse diagonal-block."""
    b = AffinitySSM.fixed(noise_std=0.0, rescale=False)
    ssm = b.build([(1, 16), (3, 16)], ssm_size=64)
    verse_vs_verse = ssm[:32, :32].mean().item()
    verse_vs_bridge = ssm[:32, 32:].mean().item()
    assert verse_vs_verse > verse_vs_bridge


def test_rescale_mean_near_target():
    """After rescaling, off-diagonal mean should be close to TARGET_MEAN."""
    b = AffinitySSM.fixed(noise_std=0.0, rescale=True)
    ssm = b.build([(0, 8), (1, 16), (2, 16), (1, 16), (2, 16), (5, 8)], ssm_size=64)
    diag_mask = ~torch.eye(64, dtype=torch.bool)
    off_mean = ssm[diag_mask].mean().item()
    assert abs(off_mean - AffinitySSM.TARGET_MEAN) < 0.05
