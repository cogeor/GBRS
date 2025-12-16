#!/usr/bin/env python
"""
Comprehensive test runner for GBRS.

This script runs all tests in sequence and provides a summary.
Can be used as an alternative to direct pytest invocation.
"""
import sys
import subprocess


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    """Run all test suites."""
    print("\n" + "=" * 70)
    print("GBRS Comprehensive Test Suite")
    print("=" * 70)

    all_passed = True

    # 1. Run correctness tests (baseline comparisons)
    print("\n[1/4] Correctness Tests (vs Baselines)")
    if not run_command(
        ["pytest", "tests/test_correctness.py", "-v", "-m", "baseline"],
        "Testing GBRS against sklearn baselines",
    ):
        all_passed = False
        print("❌ Correctness tests FAILED")
    else:
        print("✅ Correctness tests PASSED")

    # 2. Run convergence tests
    print("\n[2/4] Convergence Tests")
    if not run_command(
        ["pytest", "tests/test_convergence.py", "-v", "-m", "integration"],
        "Verifying training convergence on real datasets",
    ):
        all_passed = False
        print("❌ Convergence tests FAILED")
    else:
        print("✅ Convergence tests PASSED")

    # 3. Run fast performance tests
    print("\n[3/4] Performance Tests (Fast)")
    if not run_command(
        ["pytest", "tests/test_performance.py", "-v", "-k", "1000"],
        "Running performance benchmarks (1K samples)",
    ):
        all_passed = False
        print("❌ Performance tests FAILED")
    else:
        print("✅ Performance tests PASSED")

    # 4. Run integration tests
    print("\n[4/4] Legacy Integration Tests")
    if not run_command(
        ["python", "tests/test_integration.py"], "Running legacy integration tests"
    ):
        all_passed = False
        print("❌ Integration tests FAILED")
    else:
        print("✅ Integration tests PASSED")

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
