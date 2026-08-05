from two_brain_router.privacy import PIIGuard, assert_masked_token_invariant


def test_mask_rehydrate_round_trip():
    guard = PIIGuard()
    original = "Contact jane.doe@example.com or 555-123-4567, SSN 123-45-6789."
    result = guard.mask(original)
    assert "jane.doe@example.com" not in result.masked_text
    assert "555-123-4567" not in result.masked_text
    assert "123-45-6789" not in result.masked_text
    assert_masked_token_invariant(original, result)
    assert guard.rehydrate(result.masked_text, result.vault) == original


def test_mask_no_pii_is_noop():
    guard = PIIGuard()
    original = "What time zone is Tokyo in?"
    result = guard.mask(original)
    assert result.masked_text == original
    assert result.vault == {}
