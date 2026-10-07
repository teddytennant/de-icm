import itertools

import numpy as np
import pytest

from de_icm import de_icm, phi, phi_prime


def exact_placement(stacks, m=None):
    """Enumerate finishing orders. P(pi) = prod_k s[pi_k] / sum(s[pi_k:])."""
    s = np.asarray(stacks, dtype=np.float64)
    n = s.size
    if m is None:
        m = n
    Q = np.zeros((n, m), dtype=np.float64)
    for pi in itertools.permutations(range(n)):
        prob = 1.0
        for k in range(n):
            prob *= s[pi[k]] / np.sum(s[list(pi[k:])])
        for place, player in enumerate(pi[:m]):
            Q[player, place] += prob
    return Q


def test_phi_at_zero():
    assert phi(np.array(0.0)) == pytest.approx(np.exp(-1.0))
    assert phi_prime(np.array(0.0)) == pytest.approx(2.0 * np.exp(-1.0))


def test_two_players_match_stack_shares():
    Q, _ = de_icm([1.0, 3.0])
    assert Q[:, 0] == pytest.approx([0.25, 0.75], abs=1e-8)
    assert Q[:, 1] == pytest.approx(1.0 - Q[:, 0], abs=1e-12)


def test_equal_stacks_are_uniform():
    Q, _ = de_icm(np.ones(5))
    assert Q == pytest.approx(np.full((5, 5), 0.2), abs=1e-8)


def test_exact_enumeration_n4():
    stacks = np.array([1.0, 2.0, 4.0, 8.0])
    Q, _ = de_icm(stacks)
    ref = exact_placement(stacks)
    assert np.max(np.abs(Q - ref)) < 1e-7


def test_exact_enumeration_n6():
    rng = np.random.default_rng(0)
    stacks = rng.uniform(0.5, 5.0, size=6)
    Q, _ = de_icm(stacks)
    ref = exact_placement(stacks)
    err = np.max(np.abs(Q - ref))
    assert err < 1e-6, err


def test_row_and_column_sums():
    Q, _ = de_icm([1.0, 2.0, 5.0, 9.0])
    assert Q.sum(axis=0) == pytest.approx(np.ones(4), abs=1e-8)
    assert Q.sum(axis=1) == pytest.approx(np.ones(4), abs=1e-8)


def test_scale_invariance():
    s = np.array([1.0, 3.0, 4.0, 8.0])
    Q, _ = de_icm(s)
    Q2, _ = de_icm(2.0 * s)
    assert Q2 == pytest.approx(Q, abs=1e-12)


def test_permutation_equivariance():
    s = np.array([1.0, 4.0, 2.0, 7.0])
    perm = np.array([2, 0, 3, 1])
    Q, _ = de_icm(s)
    Qp, _ = de_icm(s[perm])
    assert Qp == pytest.approx(Q[perm], abs=1e-12)


def test_winner_take_all_equity():
    s = np.array([1.0, 2.0, 4.0, 8.0])
    prizes = np.array([1.0, 0.0, 0.0, 0.0])
    Q, V = de_icm(s, prizes)
    shares = s / s.sum()
    assert V == pytest.approx(Q[:, 0], abs=1e-15)
    assert Q[:, 0] == pytest.approx(shares, abs=1e-8)


def test_larger_stack_wins_first_more_often():
    Q, _ = de_icm([1.0, 2.0, 10.0])
    assert Q[2, 0] > Q[1, 0] > Q[0, 0]
