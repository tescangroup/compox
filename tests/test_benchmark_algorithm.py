"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from tests.test_utils import (
    is_valid_uuid,
    benchmark_algorithm,
    get_algorithm_id,
)


# Test 1: Basic positive benchmark request
def test_basic_positive_benchmark(server_url):
    base_url = f"{server_url}/api/v0/benchmark-algorithm"
    algorithm_url = f"{server_url}/api/v0/algorithm"

    response = get_algorithm_id(algorithm_url, "foo", "1")
    print(response.json())
    assert response.status_code == 200
    algorithm_id = response.json()["algorithm_id"]

    response = benchmark_algorithm(base_url, algorithm_id)
    print(response.json())
    assert response.status_code == 200
    assert "benchmark_id" in response.json()
    assert is_valid_uuid(response.json()["benchmark_id"])


# Test 2: Invalid algorithm id returns 404
def test_benchmark_invalid_algorithm_id(server_url):
    base_url = f"{server_url}/api/v0/benchmark-algorithm"

    response = benchmark_algorithm(base_url, "invalid-algorithm-id")
    print(response.json())
    assert response.status_code == 404
    assert "detail" in response.json()


# Test 3: Missing algorithm_id returns 422
def test_benchmark_missing_algorithm_id(server_url):
    base_url = f"{server_url}/api/v0/benchmark-algorithm"

    response = benchmark_algorithm(base_url)
    print(response.json())
    assert response.status_code == 422
    assert "detail" in response.json()
