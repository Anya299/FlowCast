"""
FlowCast Day 11: Reusable Python Functions.
"""

from typing import Sequence


def calculate_mae(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Calculate Mean Absolute Error."""
    if len(actual) == 0:
        raise ValueError("Input sequences cannot be empty.")

    if len(actual) != len(predicted):
        raise ValueError("Input sequences must have equal lengths.")

    errors = [
        abs(a - p)
        for a, p in zip(actual, predicted)
    ]

    return sum(errors) / len(errors)


def calculate_rmse(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Calculate Root Mean Squared Error."""
    if len(actual) == 0:
        raise ValueError("Input sequences cannot be empty.")

    if len(actual) != len(predicted):
        raise ValueError("Input sequences must have equal lengths.")

    squared_errors = [
        (a - p) ** 2
        for a, p in zip(actual, predicted)
    ]

    return (sum(squared_errors) / len(squared_errors)) ** 0.5


def is_prime(number: int) -> bool:
    """Return True if number is prime."""
    if number < 2:
        return False

    divisor = 2

    while divisor * divisor <= number:
        if number % divisor == 0:
            return False

        divisor += 1

    return True


def top_k(values: Sequence[float], k: int) -> list[float]:
    """Return the k largest values in descending order."""
    if k < 0:
        raise ValueError("k cannot be negative.")

    if k > len(values):
        raise ValueError("k cannot exceed the number of values.")

    return sorted(values, reverse=True)[:k]


def main() -> None:
    actual = [100, 120, 150, 180]
    predicted = [110, 115, 140, 200]

    print("FLOWCAST DAY 11")
    print("-" * 35)

    print("MAE:", calculate_mae(actual, predicted))
    print("RMSE:", calculate_rmse(actual, predicted))

    print("Is 17 prime?", is_prime(17))
    print("Is 18 prime?", is_prime(18))

    traffic_volumes = [120, 450, 230, 890, 340]
    print("Top 3 traffic volumes:", top_k(traffic_volumes, 3))


if __name__ == "__main__":
    main()