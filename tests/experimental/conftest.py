# SPDX-License-Identifier: Apache-2.0
"""Marker registration of tests/experimental (kept minimal: no fixtures, nothing imported)."""


def pytest_configure(config):
    config.addinivalue_line("markers", "arch_rubin: needs a Rubin (SM107, compute capability 10.7) GPU; the test is SKIPPED on other architectures")
