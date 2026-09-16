"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from tests.test_utils import get_algorithm_id


# Test 1: Basic Positive Test
def test_basic_positive_get_algorithm(server_url):
    base_url = f"{server_url}/api/v0/algorithm"
    response = get_algorithm_id(base_url, "foo", "1")
    print(response.json())
    assert response.status_code == 200
    assert "algorithm_id" in response.json()
    assert "algorithm_name" in response.json()
    assert "algorithm_version" in response.json()
    assert "storage_metrics" in response.json()
    assert "logical_size_bytes" in response.json()["storage_metrics"]
    assert "latest_algorithm_minor_version" in response.json()
    assert "algorithm_minor_version" in response.json()
    assert (
        response.json()["algorithm_minor_version"]
        == response.json()["latest_algorithm_minor_version"]
    )


# Test 2: Algorithm Name Case Sensitivity
def test_case_sensitivity(server_url):
    base_url = f"{server_url}/api/v0/algorithm"
    response = get_algorithm_id(base_url, "Foo", "1")
    print(response.json())
    assert response.status_code == 200
    assert "algorithm_id" in response.json()
    assert "algorithm_name" in response.json()
    assert "algorithm_version" in response.json()


# Test 3: Non-Existent Algorithm Name
def test_non_existent_name(server_url):
    base_url = f"{server_url}/api/v0/algorithm"
    response = get_algorithm_id(base_url, "NoSuchAlgorithm", "1")
    print(response.json())
    assert response.status_code == 404
    assert "detail" in response.json()


# Test 4: Non-Existent Algorithm Version
def test_non_existent_version(server_url):
    base_url = f"{server_url}/api/v0/algorithm"
    response = get_algorithm_id(base_url, "foo", "999")
    print(response.json())
    assert response.status_code == 404
    assert "detail" in response.json()


# Test 5: Missing Algorithm Name
def test_missing_name(server_url):
    base_url = f"{server_url}/api/v0/algorithm"
    response = get_algorithm_id(base_url, "foo", "1", use_name=False)
    print(response.json())
    assert response.status_code == 404
    assert "detail" in response.json()


# Test 6: Missing Algorithm Version
def test_missing_version(server_url):
    base_url = f"{server_url}/api/v0/algorithm"
    response = get_algorithm_id(base_url, "foo", "1", use_version=False)
    print(response.json())
    assert response.status_code == 404
    assert "detail" in response.json()


# Test 7: Missing Algorithm Name and Version
def test_missing_name_and_version(server_url):
    base_url = f"{server_url}/api/v0/algorithm"
    response = get_algorithm_id(
        base_url, "foo", "1", use_name=False, use_version=False
    )
    print(response.json())
    assert response.status_code == 404
    assert "detail" in response.json()


# Test 8: Test multiple requests returning the same algorithm
def test_multiple_requests(server_url):
    base_url = f"{server_url}/api/v0/algorithm"
    n = 10
    prev_response = None
    for i in range(n):
        response = get_algorithm_id(base_url, "foo", "1")
        print(response.json())
        assert response.status_code == 200
        assert "algorithm_id" in response.json()
        assert "algorithm_name" in response.json()
        assert "algorithm_version" in response.json()
        if i > 0:
            assert response.json() == prev_response.json()
        prev_response = response
