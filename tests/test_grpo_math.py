"""Tests for the torch-free GRPO math (kgat.train.grpo pure helpers)."""

from __future__ import annotations

import math

from kgat.train.grpo import compute_advantages


def test_dr_grpo_advantages_are_centered_not_scaled():
    rewards = [1.0, 0.0, 0.5]
    adv = compute_advantages(rewards, scale_rewards=False)
    assert math.isclose(sum(adv), 0.0, abs_tol=1e-12)  # centered
    assert math.isclose(adv[0], 0.5) and math.isclose(adv[1], -0.5)
    # No std division: spread reflects the raw reward spread.
    assert math.isclose(max(adv) - min(adv), 1.0)


def test_legacy_zscore_upweights_low_variance_groups():
    # The documented difficulty bias: a tight group gets the SAME advantage
    # magnitude as a wide group once z-scored, despite 10x less reward spread.
    wide = compute_advantages([1.0, 0.0], scale_rewards=True)
    tight = compute_advantages([0.55, 0.45], scale_rewards=True)
    assert math.isclose(abs(wide[0]), abs(tight[0]), rel_tol=1e-4)
    # The Dr. GRPO fix keeps them proportional to the actual spread.
    wide_fix = compute_advantages([1.0, 0.0], scale_rewards=False)
    tight_fix = compute_advantages([0.55, 0.45], scale_rewards=False)
    assert abs(wide_fix[0]) > abs(tight_fix[0]) * 5


def test_uniform_group_gives_zero_advantages():
    for scale in (True, False):
        adv = compute_advantages([0.7, 0.7, 0.7], scale_rewards=scale)
        assert all(abs(a) < 1e-9 for a in adv)


def test_empty_group():
    assert compute_advantages([], scale_rewards=False) == []


def test_zscore_normalizes_to_unit_std():
    rewards = [2.0, 4.0, 6.0, 8.0]
    adv = compute_advantages(rewards, scale_rewards=True)
    n = len(adv)
    std = (sum(a * a for a in adv) / n) ** 0.5
    assert math.isclose(std, 1.0, rel_tol=1e-3)
