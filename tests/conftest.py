import pytest
import torch


@pytest.fixture
def piano_roll():
    """64 beats, 128 notes — C note every 4 beats."""
    roll = torch.zeros(64, 128)
    roll[::4, 60] = 1
    return roll


@pytest.fixture
def segment_plan():
    """[(label_id, n_bars)] — intro:8, verse:16, chorus:16, outro:8"""
    return [(0, 8), (1, 16), (2, 16), (5, 8)]


@pytest.fixture
def segment_plan_str():
    return "intro:8,verse:16,chorus:16,outro:8"
