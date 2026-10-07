"""Root runner. The checks live in tests/test_risk_calibration.py."""

if __name__ == "__main__":
    import tests.test_risk_calibration as checks

    checks.test_metrics_use_debt_multiple()
    checks.test_debt_units()
    checks.test_profile_order()
    checks.test_outside_profile_is_not_sell()
    checks.test_sector_cap_is_shared()
    checks.test_quote_skips_rejection_rate()
    print("risk calibration ok")
