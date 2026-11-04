# ── simulatedParts uses null, not absence ────────────────────────────────────


def _manifest(parts):
    from gensyn_audit.record import RecordManifest

    return RecordManifest(
        run_id="r",
        run_name="open-1b",
        total_steps=None,
        steps_per_segment=None,
        artifact_check=None,
        claim_ttl_hours=None,
        exchange_format=None,
        simulated=True,
        simulated_parts=parts,
        endpoints={},
    )


def test_null_part_means_that_part_is_real():
    # The record lists every part as a key: a string explains what is simulated,
    # null means it is genuine. Testing for the key alone fired the loudest
    # warning this tool has against a real committed hash.
    m = _manifest({"commitments": None, "lossGate": "not wired up"})
    assert m.commitments_are_placeholders is False
    assert m.loss_gate_is_live is False


def test_a_message_means_that_part_is_simulated():
    m = _manifest({"commitments": "derived placeholders", "lossGate": None})
    assert m.commitments_are_placeholders is True
    assert m.loss_gate_is_live is True


def test_the_codes_the_record_actually_returns_are_explained():
    """`no_token` and `bad_request` are both live on production and were
    surfacing as bare codes. The first almost always means a missing --claim."""
    from gensyn_audit.record import _ERROR_GUIDANCE

    assert "--claim" in _ERROR_GUIDANCE["no_token"]
    assert "not your" in _ERROR_GUIDANCE["bad_request"]


# ── a run that is still training ─────────────────────────────────────────────


def _refusal(status: int, body: dict, method: str = "GET", path: str = "/v1/runs/open-8b/x"):
    import io
    import json
    import urllib.error

    from gensyn_audit.record import _api_error

    raw = io.BytesIO(json.dumps(body).encode())
    return _api_error(urllib.error.HTTPError("u", status, "", {}, raw), method, path)


def test_a_step_past_the_published_segments_says_so_rather_than_refused():
    """open-8b opens for audit while it trains. Past `publishedSegments` the
    record answers 404 (receipt, segment, anchor) or 409 (claim) with
    `not_published`; that is a wait, not a broken request or a bad token."""
    err = _refusal(
        404,
        {
            "error": "not_published",
            "detail": "That segment is not published yet: the run is still training.",
            "segment": 2,
            "publishedSegments": 2,
        },
        path="/v1/runs/open-8b/steps/1000/receipt",
    )
    assert (
        str(err) == "that step is not open to audit yet: segment 2 is past the 2 published so far."
    )
    assert "still training" in (err.hint or "")
    assert "the run is still training." in (err.hint or "")
    assert "refused" not in str(err)

    claim = _refusal(409, {"error": "not_published", "detail": "x"}, method="POST")
    assert str(claim) == "that step is not open to audit yet."
    assert (claim.hint or "").endswith("\nx")


def test_other_refusals_keep_their_shape():
    err = _refusal(401, {"error": "bad_token", "detail": "no"})
    assert str(err) == "GET /v1/runs/open-8b/x was refused (401 bad_token)."
