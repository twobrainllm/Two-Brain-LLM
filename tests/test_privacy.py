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


def test_known_gap_person_names_are_not_masked():
    """Pins a REAL, currently-unfixed hole in the privacy guarantee.

    `privacy/patterns.py` only covers regex-shaped identifiers (EMAIL, PHONE,
    SSN, CREDIT_CARD). Person names are an open class and are **not masked**,
    so a name escapes `mask()` untouched.

    This was harmless while `CloudDeepBrain` was a stub. It is not harmless now
    that `CirrascaleDeepBrain` sends masked text to a real third-party
    endpoint: a first real escalation test sent "Sarah Chen" off-device in
    plain text. See data/cloud_ai100/_real_endpoint_log.md.

    This test asserts the *current, broken* behaviour on purpose, so the gap is
    visible in the suite instead of hidden behind
    `test_escalated_pii_never_reaches_cloud_unmasked`, which only ever checks
    an email. **When names are handled, this test should start failing** --
    that is the signal to invert it, not to delete it.
    """
    guard = PIIGuard()
    result = guard.mask("My name is Sarah Chen and my email is sarah@acme.com")

    assert "[PII_EMAIL_1]" in result.masked_text, "email should be masked"
    assert "Sarah Chen" in result.masked_text, (
        "EXPECTED FAILURE ONCE FIXED: person names are still unmasked. "
        "If this assertion fails, name masking now works -- invert this test."
    )
