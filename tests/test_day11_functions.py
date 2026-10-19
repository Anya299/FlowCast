import pytest

from src.day11_functions import (
    calculate_mae,
    calculate_rmse,
    is_prime,
    top_k,
)


def test_calculate_mae():
    actual = [100, 120, 150, 180]
    predicted = [110, 115, 140, 200]

    assert calculate_mae(actual, predicted) == 11.25


def test_calculate_rmse():
    actual = [100, 120, 150, 180]
    predicted = [110, 115, 140, 200]

    assert calculate_rmse(actual, predicted) == pytest.approx(12.5)


def test_mae_rejects_empty_inputs():
    with pytest.raises(ValueError):
        calculate_mae([], [])


def test_metrics_reject_different_lengths():
    with pytest.raises(ValueError):
        calculate_mae([100, 200], [110])


def test_is_prime():
    assert is_prime(2) is True
    assert is_prime(17) is True
    assert is_prime(18) is False
    assert is_prime(1) is False
    assert is_prime(-5) is False


def test_top_k():
    values = [120, 450, 230, 890, 340]

    assert top_k(values, 3) == [890, 450, 340]


def test_top_k_zero():
    assert top_k([120, 450], 0) == []


def test_top_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        top_k([100, 200], 3)

    with pytest.raises(ValueError):
        top_k([100, 200], -1)