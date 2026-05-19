import torch
import pytest

from pipeline.generate import parse_segment_plan, build_ssm, SSMType, LABEL_NAME_TO_ID


def test_parse_basic():
    result = parse_segment_plan("verse:16,chorus:16")
    assert result == [
        (LABEL_NAME_TO_ID["verse"], 16),
        (LABEL_NAME_TO_ID["chorus"], 16),
    ]


def test_parse_full_plan(segment_plan_str):
    result = parse_segment_plan(segment_plan_str)
    assert len(result) == 4
    assert result[0] == (LABEL_NAME_TO_ID["intro"], 8)
    assert result[-1] == (LABEL_NAME_TO_ID["outro"], 8)


def test_parse_empty_raises():
    with pytest.raises((ValueError, Exception)):
        parse_segment_plan("")


def test_parse_unknown_label_raises():
    with pytest.raises(ValueError):
        parse_segment_plan("unknown_section:8")


def test_build_ssm_none_is_zeros(segment_plan):
    ssm = build_ssm(SSMType.NONE, segment_plan, total_len=32)
    assert ssm.shape == (32, 32)
    assert ssm.sum().item() == 0.0


def test_build_ssm_random_symmetric(segment_plan):
    ssm = build_ssm(SSMType.RANDOM, segment_plan, total_len=32)
    assert ssm.shape == (32, 32)
    assert torch.allclose(ssm, ssm.T, atol=1e-5)
    assert ssm.min() >= 0.0
    assert ssm.max() <= 1.0
