"""What must hold about the checkpoint an audit STARTS from.

Two predecessors, two different claims, and the tool has to keep them apart:

* an original Gensyn checkpoint has no gradients and never will, so its
  training-state hash cannot be reconstructed. Its bytes are checked against
  the digests the record publishes, and the result says in words that the state
  inside them is assumed, not proved.
* another auditor's hand-off has gradients precisely so that it can be proved,
  and everything about it fails closed: no gradients, no rank chains, a forged
  claim of being a Gensyn anchor, a hash that does not reconstruct — none of
  them degrade to the anchor path.

Driven through `gensyn-audit verify`, which is the same gate `run` applies
before it commits a machine to a replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import write_handoff

from gensyn_audit import cli

RUN = "20260703-171943-4e85cd3"
REPOP_SHA = "244c0791e378a180dd6b29bbf8f2e244b2bea806"
KIT_ID = f"pt-{'a' * 12}_rp-{REPOP_SHA[:12]}"

#: Log steps 98, 99 and 100 are hashed; 100 is the predecessor of audit step
#: 100 (log step 101), which is what `--step 100` audits.
PUBLISHED = {98: "8" * 64, 99: "9" * 64, 100: "1" * 64, 101: "2" * 64}

#: What the stub verifier reconstructs. Matching `PUBLISHED[100]` is a pass.
_VERIFIER = """#!/bin/sh
EXPECT=""; PREV=""; CKPT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --expect-hash) EXPECT="$2"; shift 2;;
    --prev-hash) PREV="$2"; shift 2;;
    --checkpoint) CKPT="$2"; shift 2;;
    *) shift;;
  esac
done
# Stands in for the pinned pretrain: the reconstruction is tested there
# (tests/test_verify_handoff.py). What matters here is that the CLI hands it
# the PUBLISHED hashes and refuses whatever it says no to.
GOT="__RECONSTRUCTS__"
[ -f "$CKPT/gradients.safetensors" ] || GOT="0000000000000000000000000000000000000000000000000000000000000000"
echo "{\\"checkpoint\\": \\"$CKPT\\", \\"expected_hash\\": \\"$EXPECT\\", \\"prev_hash\\": \\"$PREV\\",
  \\"reconstructed_hash\\": \\"$GOT\\", \\"dp_world_size\\": 1, \\"step\\": 100,
  \\"verified\\": $([ "$GOT" = "$EXPECT" ] && echo true || echo false)}"
"""


def _artifact_files(root: Path) -> list[dict]:
    return [
        {
            "name": str(p.relative_to(root)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A kit, a venv holding the verifier, a published log, and a record."""
    monkeypatch.setenv("AUDIT_CACHE", str(tmp_path / "cache"))

    kit_dir = tmp_path / "kit"
    kit_dir.mkdir()
    traj = {
        "trajectory_format": 1,
        "name": "t",
        "repop_commit": REPOP_SHA,
        "pretrain_commit": "a" * 40,
        "notes": "",
        "units": [
            {
                "kind": "init",
                "config_name": "1b_repop_run3",
                "until_step": 0,
                "state_hash": "e" * 64,
                "devices_verified": ["cpu"],
            }
        ],
    }
    tb = (json.dumps(traj) + "\n").encode()
    (kit_dir / "trajectory.json").write_bytes(tb)
    (kit_dir / "kit.json").write_text(
        json.dumps(
            {
                "kit_format": 1,
                "pretrain_commit": "a" * 40,
                "repop_commit": REPOP_SHA,
                "files": [
                    {
                        "name": "trajectory.json",
                        "sha256": hashlib.sha256(tb).hexdigest(),
                        "bytes": len(tb),
                    }
                ],
            }
        )
    )

    venv_bin = tmp_path / "cache" / "kits" / KIT_ID / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    verifier = venv_bin / "pretrain-audit-verify-handoff"
    verifier.write_text(_VERIFIER.replace("__RECONSTRUCTS__", PUBLISHED[100]))
    verifier.chmod(0o755)

    log = tmp_path / "state_hashes.jsonl"
    log.write_text(
        "".join(
            json.dumps({"step": s, "state_hash": h}) + "\n" for s, h in sorted(PUBLISHED.items())
        )
    )

    return {"kit": kit_dir, "tmp": tmp_path, "log": log, "venv_bin": venv_bin}


def _fixture(
    world, pred_dir: Path, *, kind: str, digests: bool = True, extra: dict | None = None
) -> Path:
    doc = {
        "run": RUN,
        "steps": {
            "100": {
                "committed_hash": PUBLISHED[101],
                "predecessor": {
                    "source": kind,
                    "step": 100,
                    "uri": str(pred_dir),
                    "digest": PUBLISHED[100],
                    **({"artifactFiles": _artifact_files(pred_dir)} if digests else {}),
                    **(extra or {}),
                },
            }
        },
    }
    path = world["tmp"] / f"fixture-{kind}-{digests}.json"
    path.write_text(json.dumps(doc))
    return path


def _verify(world, fixture: Path, *extra) -> int:
    return cli.main(
        [
            "verify",
            "--kit",
            str(world["kit"]),
            "--step",
            "100",
            "--run",
            RUN,
            "--record",
            f"mock://{fixture}",
            "--state-hashes",
            str(world["log"]),
            "--device",
            "cpu",
            "--workdir",
            str(world["tmp"] / "audit"),
            "--no-color",
            *extra,
        ]
    )


@pytest.mark.parametrize("fails", [False, True])
def test_standalone_verify_refreshes_the_restart_verdict_only_on_success(world, monkeypatch, fails):
    from types import SimpleNamespace

    from gensyn_audit.errors import AuditError
    from gensyn_audit.mock import MockRecord
    from gensyn_audit.plan import Workdir

    bundle = world["tmp"] / "handoff.safetensors"
    bundle.write_bytes(b"packed checkpoint")
    digest = hashlib.blake2b(bundle.read_bytes(), digest_size=32).hexdigest()
    fixture = _fixture(world, bundle, kind="crowd", digests=False, extra={"digest": digest})
    wd = Workdir(world["tmp"] / "audit")
    wd.create()
    wd.predecessor_record.write_text('{"stale": true}')

    def unpack(venv, source, dest):
        assert not wd.predecessor_record.exists(), "invalidate before replacing the directory"
        write_handoff(dest, step=100)
        if fails:
            raise AuditError("unpacking failed")
        return dest

    monkeypatch.setattr(cli.convertmod, "unpack", unpack)
    assert _verify(world, fixture) == (cli.EXIT_ERROR if fails else cli.EXIT_OK)
    assert bundle.is_file(), "standalone verify must not consume the bundle"
    ctx = MockRecord(fixture).step_context(RUN, 100)
    cached = cli._restartable_predecessor(SimpleNamespace(workdir=wd), ctx)
    if fails:
        assert not wd.predecessor_record.exists()
        assert cached is None
    else:
        saved = json.loads(wd.predecessor_record.read_text())
        assert saved["bundle_digest"] == digest
        assert saved["verdict"]["tensor_state_commitment"] == "verified"
        assert cached == wd.verified_predecessor


def _anchor(world) -> Path:
    """An original Gensyn checkpoint: a real one, so no gradients sidecar."""
    d = write_handoff(
        world["tmp"] / "anchor" / "step_000000100", step=100, omit=("gradients.safetensors",)
    )
    (d / "state_hash.txt").write_text(PUBLISHED[100] + "\n")
    return d


def _crowd(w, **kw) -> Path:
    d = write_handoff(w["tmp"] / "crowd" / "step_000000100", step=100, **kw)
    (d / "state_hash.txt").write_text(PUBLISHED[100] + "\n")
    return d


# ── 1. the trusted anchor path ───────────────────────────────────────────────


def test_a_trusted_anchor_without_gradients_proceeds(world, capsys):
    fixture = _fixture(world, _anchor(world), kind="published")

    assert _verify(world, fixture) == 0
    out = capsys.readouterr().out

    assert "Trusted Gensyn anchor" in out
    assert "artifact integrity" not in out
    assert "training-state hash" not in out


def test_a_modified_anchor_fails_before_anything_opens_it(world, capsys):
    """The digest check is the whole of an anchor's trust, so it has to be the
    thing that stops — and it has to stop before the dcp/.metadata pickle."""
    pred = _anchor(world)
    fixture = _fixture(world, pred, kind="published")
    (pred / "dcp" / "__0_0.distcp").write_bytes(b"tampered")

    assert _verify(world, fixture) == cli.EXIT_ERROR
    out = capsys.readouterr().out + capsys.readouterr().err
    del out


def test_an_extra_file_in_an_anchor_is_a_modification(world):
    pred = _anchor(world)
    fixture = _fixture(world, pred, kind="published")
    (pred / "surprise.pt").write_text("hello")

    assert _verify(world, fixture) == cli.EXIT_ERROR


def test_a_digestless_anchor_is_trusted_without_warning(world, capsys):
    fixture = _fixture(world, _anchor(world), kind="published", digests=False)

    assert _verify(world, fixture) == 0
    both = capsys.readouterr()
    assert "Trusted Gensyn anchor" in both.out
    assert "UNVERIFIED" not in both.out + both.err


# ── 2. the crowd path ────────────────────────────────────────────────────────


def test_a_crowd_handoff_with_gradients_passes_the_full_state_hash_gate(world, capsys):
    fixture = _fixture(world, _crowd(world), kind="crowd-provided checkpoint")

    assert _verify(world, fixture) == 0
    out = capsys.readouterr().out

    assert "tensor state reconstructed and matched" in out
    assert "reconstructed and matched against the published log" in out
    assert "not independently verified" not in out


def test_the_gate_is_given_the_published_hashes_not_the_artifact_author_s(world):
    """`--expect-hash` is the published log's entry for the predecessor's step
    and `--prev-hash` is the previous HASHED step's. Neither is read out of the
    checkpoint, whose author would otherwise be grading their own work."""
    pred = _crowd(world)
    (pred / "state_hash.txt").write_text("f" * 64 + "\n")  # the author's claim
    fixture = _fixture(world, pred, kind="crowd-provided checkpoint")

    argv = world["tmp"] / "argv.txt"
    verifier = world["venv_bin"] / "pretrain-audit-verify-handoff"
    verifier.write_text(
        _VERIFIER.replace("__RECONSTRUCTS__", PUBLISHED[100]).replace(
            'EXPECT=""', f'echo "$@" > {argv}\nEXPECT=""', 1
        )
    )
    verifier.chmod(0o755)

    assert _verify(world, fixture) == 0
    passed = argv.read_text()
    assert f"--expect-hash {PUBLISHED[100]}" in passed
    assert f"--prev-hash {PUBLISHED[99]}" in passed
    assert "f" * 64 not in passed, "the artifact author's own claim never reaches the gate"


def test_a_crowd_handoff_without_gradients_fails_and_does_not_fall_back(world, capsys):
    fixture = _fixture(
        world, _crowd(world, omit=("gradients.safetensors",)), kind="crowd-provided checkpoint"
    )

    assert _verify(world, fixture) == cli.EXIT_ERROR
    both = capsys.readouterr()
    assert "gradients.safetensors" in both.err
    assert "Trusted Gensyn anchor" not in both.out, "must never become an anchor"


def test_a_crowd_handoff_missing_a_rank_chain_fails(world, capsys):
    fixture = _fixture(
        world,
        _crowd(world, world=2, omit=("batch_hasher.rank_1.bin",)),
        kind="crowd-provided checkpoint",
    )

    assert _verify(world, fixture) == cli.EXIT_ERROR
    assert "batch_hasher.rank_1.bin" in capsys.readouterr().err


def test_forged_gensyn_metadata_inside_the_artifact_changes_nothing(world, capsys):
    """Provenance comes from the record. A hand-off that writes `published`
    into its own meta.json is still a crowd hand-off, and still fails closed."""
    pred = _crowd(world, omit=("gradients.safetensors",))
    (pred / "meta.json").write_text(
        json.dumps(
            {
                "step": 100,
                "dp_world_size": 1,
                "chained_hash": PUBLISHED[100],
                "kind": "published",
                "source": "gensyn",
                "gensyn": {"anchor": True},
            }
        )
    )
    fixture = _fixture(world, pred, kind="crowd-provided checkpoint")

    assert _verify(world, fixture) == cli.EXIT_ERROR
    assert "Trusted Gensyn anchor" not in capsys.readouterr().out


def test_a_crowd_handoff_whose_state_does_not_reconstruct_fails(world, capsys):
    """The stub reconstructs the published hash for step 100; point the record
    at a step whose published hash is a different one and it must refuse."""
    verifier = world["venv_bin"] / "pretrain-audit-verify-handoff"
    verifier.write_text(_VERIFIER.replace("__RECONSTRUCTS__", "c" * 64))
    verifier.chmod(0o755)
    fixture = _fixture(world, _crowd(world), kind="crowd-provided checkpoint")

    assert _verify(world, fixture) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "not the state the run committed to" in err
    assert "cccccccc" in err and PUBLISHED[100][:8] in err


def test_a_kit_whose_pretrain_predates_the_verifier_refuses_rather_than_skips(world, capsys):
    (world["venv_bin"] / "pretrain-audit-verify-handoff").unlink()
    fixture = _fixture(world, _crowd(world), kind="crowd-provided checkpoint")

    assert _verify(world, fixture) == cli.EXIT_ERROR
    assert "must not be replayed from unchecked" in capsys.readouterr().err


def test_without_a_published_log_a_crowd_handoff_cannot_be_verified(world, capsys):
    fixture = _fixture(world, _crowd(world), kind="crowd-provided checkpoint")
    code = cli.main(
        [
            "verify",
            "--kit",
            str(world["kit"]),
            "--step",
            "100",
            "--run",
            RUN,
            "--record",
            f"mock://{fixture}",
            "--device",
            "cpu",
            "--no-color",
        ]
    )
    assert code == cli.EXIT_ERROR
    assert "artifacts.state_hashes" in capsys.readouterr().err


# ── 3. provenance the record does not state ──────────────────────────────────


@pytest.mark.parametrize("kind", ["", "mirror", "trusted", "gensyn.ai"])
def test_provenance_the_record_does_not_state_fails_closed(world, capsys, kind):
    """Criterion: unknown provenance cannot silently bypass verification.
    Defaulting to the anchor path would make the anchor path the bypass."""
    fixture = _fixture(world, _crowd(world), kind=kind)

    assert _verify(world, fixture) == cli.EXIT_ERROR
    assert "cannot be classified" in capsys.readouterr().err


# ── 4. step numbering, which is where this goes wrong quietly ────────────────


def test_the_published_hashes_are_looked_up_in_log_numbering(world, capsys):
    """`--step 100` is the AUDIT number: it audits log step 101 starting from
    the log-step-100 checkpoint. The gate must ask the log for 100 and for the
    previous HASHED step, 99 — not 99 and 98, and not 101 and 100."""
    fixture = _fixture(world, _crowd(world), kind="crowd-provided checkpoint")

    assert _verify(world, fixture) == 0
    out = capsys.readouterr().out
    assert f"{PUBLISHED[100]}" in out and "log step 100" in out
    assert PUBLISHED[99][:16] in out and "log step 99" in out


def test_the_preceding_hash_is_the_previous_hashed_step_not_step_minus_one(world):
    """With a cadence of 2 the value the chain folds in is two steps back. The
    published log is the definition of which steps were hashed, so nothing here
    does the arithmetic."""
    from gensyn_audit.commitments import Commitments

    c = Commitments(source="x", by_step={90: "a" * 64, 92: "b" * 64, 94: "c" * 64})
    assert c.previous(94) == (92, "b" * 64)
    assert c.previous(92) == (90, "a" * 64)


def test_a_log_that_starts_after_the_step_is_the_wrong_log():
    from gensyn_audit.commitments import Commitments
    from gensyn_audit.errors import AuditError

    c = Commitments(source="x", by_step={200: "a" * 64})
    with pytest.raises(AuditError, match="no hash before step 200"):
        c.previous(200)


# ── 5. what the gate must not touch ──────────────────────────────────────────


def test_an_init_unit_has_no_predecessor_and_no_gate(world, capsys):
    """Init verification regenerates the state from a seed. It reads no
    checkpoint and needs no sidecar, and this work must not have given it one."""
    code = cli.main(
        [
            "plan",
            "--kit",
            str(world["kit"]),
            "--config-name",
            "1b_repop_run3",
            "--kind",
            "init",
            "--device",
            "cpu",
            "--no-color",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "--from-init" in out and "--until-step 0" in out
    assert "gradients" not in out and "predecessor" not in out


# ── 6. transport: the sidecars have to make the trip ─────────────────────────


def _staged_remote(monkeypatch, tmp_path, names) -> None:
    """Stand a GCS prefix up from a list of object names."""
    from gensyn_audit import gcs

    objects = [{"name": f"p/{n}", "size": len(n), "md5": None, "crc32c": None} for n in names]
    monkeypatch.setattr(gcs, "list_prefix", lambda uri, creds=None: objects)
    monkeypatch.setattr(gcs, "credentials", lambda: gcs.Credentials("anonymous"))

    def fake_download(uri, dest, creds=None, *, expected=None, on_progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps({"step": 100, "dp_world_size": 2}) if dest.name == "meta.json" else dest.name
        )
        if on_progress:
            on_progress(len(dest.name), len(dest.name))
        return dest

    monkeypatch.setattr(gcs, "download", fake_download)


_MEMBERS = (
    "meta.json",
    "_COMPLETE",
    "state_hash.txt",
    "global_stream.json",
    "rng.rank_0.pt",
    "gradients.safetensors",
    "batch_hasher.rank_0.bin",
    "batch_hasher.rank_1.bin",
    "dcp/__0_0.distcp",
)


def test_a_downloaded_handoff_brings_its_gradients_and_rank_chains(monkeypatch, tmp_path):
    """Not "they exist where the replay wrote them" — they arrive at the other
    end. `download_prefix` mirrors the whole prefix, and this is the assertion
    that keeps it that way."""
    from gensyn_audit import fetch
    from gensyn_audit import handoff as handoff_mod
    from gensyn_audit.kit import Unit

    _staged_remote(monkeypatch, tmp_path, _MEMBERS)
    dest = tmp_path / "pred"
    unit = Unit(
        kind="interval",
        config_name="c",
        until_step=101,
        state_hash="a" * 64,
        devices_verified=(),
        mps_wall_seconds=None,
        mps_peak_rss_gb=None,
        checkpoint_uri="gs://b/p",
        gcs_root=None,
        expect_hash=None,
    )
    fetch.fetch_checkpoint("gs://b/p", dest, unit, expected_digest=None)

    got = handoff_mod.inspect(dest)
    assert got.is_complete, f"missing after transport: {got.missing}"
    assert got.has_gradients and got.dp_world_size == 2


def test_a_download_that_dropped_the_sidecar_is_not_treated_as_verified(
    monkeypatch, tmp_path, capsys
):
    """An artifact that arrives short is refused, not replayed from. It is the
    difference between a hand-off and a checkpoint someone vouches for."""
    from gensyn_audit import fetch
    from gensyn_audit import handoff as handoff_mod
    from gensyn_audit.errors import AuditError
    from gensyn_audit.kit import Unit

    partial = tuple(n for n in _MEMBERS if n != "gradients.safetensors")
    _staged_remote(monkeypatch, tmp_path, partial)
    dest = tmp_path / "pred"
    unit = Unit(
        kind="interval",
        config_name="c",
        until_step=101,
        state_hash="a" * 64,
        devices_verified=(),
        mps_wall_seconds=None,
        mps_peak_rss_gb=None,
        checkpoint_uri="gs://b/p",
        gcs_root=None,
        expect_hash=None,
    )

    fetch.fetch_checkpoint("gs://b/p", dest, unit, expected_digest=None)
    capsys.readouterr()

    with pytest.raises(AuditError, match="incomplete"):
        handoff_mod.require_complete(handoff_mod.inspect(dest), what="this crowd hand-off")


def test_an_unreadable_meta_is_not_a_one_rank_handoff(tmp_path):
    """The file that says how many rank chains to look for cannot be the one
    we shrug at: defaulting to 1 makes a 32-rank hand-off with a single chain
    look complete."""
    from gensyn_audit import handoff as handoff_mod
    from gensyn_audit.errors import AuditError

    d = write_handoff(tmp_path / "h")
    (d / "meta.json").write_text("{not json")
    with pytest.raises(AuditError, match="not readable JSON"):
        handoff_mod.inspect(d)


def test_crowd_without_artifact_manifest_never_invokes_verifier(world, monkeypatch):
    from gensyn_audit import verify

    fixture = _fixture(world, _crowd(world), kind="crowd-provided checkpoint", digests=False)

    def must_not_run(*args, **kwargs):
        pytest.fail("unverified crowd bytes reached the checkpoint loader")

    monkeypatch.setattr(verify, "check_state_hash", must_not_run)
    assert _verify(world, fixture) == cli.EXIT_ERROR


@pytest.mark.parametrize(
    "change,exit_code",
    [
        ({}, 1),
        ({"verified": "true"}, 0),
        ({"verified": False}, 0),
        ({"expected_hash": "f" * 64}, 0),
        ({"reconstructed_hash": "f" * 64}, 0),
        ({"prev_hash": "f" * 64}, 0),
        ({"dp_world_size": True}, 0),
    ],
)
def test_verifier_exit_and_result_must_agree(world, change, exit_code):
    import shlex

    fixture = _fixture(world, _crowd(world), kind="crowd-provided checkpoint")
    result = {
        "verified": True,
        "expected_hash": PUBLISHED[100],
        "reconstructed_hash": PUBLISHED[100],
        "prev_hash": PUBLISHED[99],
        "dp_world_size": 1,
    }
    result.update(change)
    entry = world["venv_bin"] / "pretrain-audit-verify-handoff"
    entry.write_text(
        "#!/bin/sh\nprintf '%s\\n' " + shlex.quote(json.dumps(result)) + f"\nexit {exit_code}\n"
    )
    assert _verify(world, fixture) == cli.EXIT_ERROR


@pytest.mark.parametrize("stdout", ["not JSON", "[]", "null", '{"verified": true}'])
def test_malformed_verifier_result_fails(world, stdout):
    import shlex

    fixture = _fixture(world, _crowd(world), kind="crowd-provided checkpoint")
    entry = world["venv_bin"] / "pretrain-audit-verify-handoff"
    entry.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote(stdout) + "\n")
    assert _verify(world, fixture) == cli.EXIT_ERROR


@pytest.mark.parametrize("first", [1, 10])
def test_init_hash_is_not_a_periodic_chain_link(first):
    from gensyn_audit.commitments import Commitments

    log = Commitments("log", {0: "f" * 64, first: "a" * 64})
    assert log.require(0) == "f" * 64
    assert log.previous(first) == (0, "0" * 64)


def test_step_one_without_an_init_record_has_a_zero_predecessor():
    from gensyn_audit.commitments import Commitments

    assert Commitments("log", {1: "a" * 64}).previous(1) == (0, "0" * 64)


def test_handoff_step_must_match_trusted_record(world):
    pred = _crowd(world)
    meta_path = pred / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["step"] = 99
    meta_path.write_text(json.dumps(meta))
    fixture = _fixture(world, pred, kind="crowd-provided checkpoint")
    assert _verify(world, fixture) == cli.EXIT_ERROR


def test_duplicate_artifact_manifest_entries_are_refused(world):
    pred = _anchor(world)
    entries = _artifact_files(pred)
    fixture = _fixture(
        world, pred, kind="published", extra={"artifactFiles": entries + [entries[0]]}
    )
    assert _verify(world, fixture) == cli.EXIT_ERROR


def test_symlinked_artifact_is_refused(world):
    pred = _anchor(world)
    fixture = _fixture(world, pred, kind="published")
    data = pred / "dcp" / "__0_0.distcp"
    external = world["tmp"] / "external.distcp"
    data.rename(external)
    data.symlink_to(external)
    assert _verify(world, fixture) == cli.EXIT_ERROR


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {},
        {"step": 1},
        {"step": 1, "dp_world_size": 0},
        {"step": 1, "dp_world_size": True},
        {"step": "1", "dp_world_size": 1},
    ],
)
def test_invalid_handoff_topology_is_refused(tmp_path, metadata):
    from gensyn_audit.errors import AuditError
    from gensyn_audit.handoff import inspect

    root = write_handoff(tmp_path / "handoff")
    (root / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(AuditError):
        inspect(root)


# ── 7. the contract follows what the run hashed ──────────────────────────────

#: What an open-1b checkpoint's config_resolved says (ckpt/step_000080900).
_OPEN_1B_HASH = {"every_n_steps": 1, "include_grads": True, "include_batch": True, "at_init": True}
#: What open-8b's says: weights and moments only. Its writer exports neither
#: the gradients nor the rank chains.
_OPEN_8B_HASH = {
    "every_n_steps": 1,
    "include_grads": False,
    "include_batch": False,
    "at_init": True,
}


def test_a_handoff_from_a_run_that_hashed_neither_is_complete_without_them(tmp_path):
    """open-8b: 48 ranks, no gradients sidecar, no batch chains, and nothing
    missing. Demanding files the writer never produces refuses every hand-off
    the run can make."""
    from gensyn_audit import handoff as handoff_mod

    d = write_handoff(tmp_path / "h", world=48, state_hash=_OPEN_8B_HASH)
    assert not (d / "gradients.safetensors").exists()
    assert not list(d.glob("batch_hasher.rank_*.bin"))

    h = handoff_mod.inspect(d)
    assert h.is_complete, h.missing
    assert not h.gradients_required and not h.batch_chains_required
    assert not h.has_gradients and h.dp_world_size == 48
    handoff_mod.require_complete(h, what="this hand-off")


def test_a_handoff_from_a_run_that_hashed_both_still_needs_both(tmp_path):
    """open-1b states both terms on, so dropping either one is still refused."""
    from gensyn_audit import handoff as handoff_mod
    from gensyn_audit.errors import AuditError

    full = handoff_mod.inspect(write_handoff(tmp_path / "full", world=3, state_hash=_OPEN_1B_HASH))
    assert full.is_complete and full.gradients_required and full.batch_chains_required

    for dropped in ("gradients.safetensors", "batch_hasher.rank_2.bin"):
        d = write_handoff(tmp_path / dropped, world=3, state_hash=_OPEN_1B_HASH, omit=(dropped,))
        with pytest.raises(AuditError, match="incomplete") as exc:
            handoff_mod.require_complete(handoff_mod.inspect(d), what="this hand-off")
        assert dropped in exc.value.hint


@pytest.mark.parametrize(
    "state_hash,needs",
    [
        ({"every_n_steps": 1, "include_grads": True, "include_batch": False}, "gradients"),
        ({"every_n_steps": 1, "include_grads": False, "include_batch": True}, "batch_hasher"),
    ],
)
def test_each_term_decides_its_own_member(tmp_path, state_hash, needs):
    from gensyn_audit import handoff as handoff_mod

    d = write_handoff(tmp_path / "h", world=2, state_hash=state_hash)
    for member in ("gradients.safetensors", "batch_hasher.rank_0.bin", "batch_hasher.rank_1.bin"):
        (d / member).unlink(missing_ok=True)
    missing = handoff_mod.inspect(d).missing
    assert missing and all(name.startswith(needs) for name in missing)


@pytest.mark.parametrize("include", [False, True])
def test_a_run_that_hashed_nothing_cannot_be_handed_off(tmp_path, include):
    """every_n_steps 0 publishes no commitment, so the kit's verifier refuses
    the hand-off whatever the include flags say. It must never read as a
    complete hand-off with nothing to carry."""
    from gensyn_audit import handoff as handoff_mod
    from gensyn_audit.errors import AuditError

    sh = {"every_n_steps": 0, "include_grads": include, "include_batch": include}
    d = write_handoff(tmp_path / "h", world=2, state_hash=sh)
    with pytest.raises(AuditError, match="did not hash its state"):
        handoff_mod.inspect(d)


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("step,due", [(500, True), (1000, True), (501, False), (1, False)])
def test_only_a_hash_due_step_can_be_handed_off(tmp_path, include, step, due):
    """The published log holds no hash for a step off the cadence, so the
    verifier refuses it; packing and uploading it would only be rejected."""
    from gensyn_audit import handoff as handoff_mod
    from gensyn_audit.errors import AuditError

    sh = {"every_n_steps": 100, "include_grads": include, "include_batch": include}
    d = write_handoff(tmp_path / "h", world=2, step=step, state_hash=sh)
    if due:
        assert handoff_mod.inspect(d).is_complete
    else:
        with pytest.raises(AuditError, match="off the run's hash cadence"):
            handoff_mod.inspect(d)


def test_an_unhashable_handoff_is_not_packed_or_uploaded(tmp_path):
    """The refusal reaches the two places that would otherwise send it."""
    from gensyn_audit import upload as uploadmod
    from gensyn_audit.errors import AuditError

    sh = {"every_n_steps": 100, "include_grads": False, "include_batch": False}
    d = write_handoff(tmp_path / "h", world=2, step=501, state_hash=sh, safetensors=10)
    with pytest.raises(AuditError, match="off the run's hash cadence"):
        uploadmod.build_bundle(d, tmp_path)


@pytest.mark.parametrize(
    "state_hash,grads,chains",
    [
        ({"every_n_steps": 1, "include_batch": False}, True, False),
        ({"every_n_steps": 1, "include_grads": False}, False, True),
        ({"every_n_steps": 1}, True, True),
        ({}, True, True),
    ],
)
def test_an_unstated_term_keeps_the_full_contract(tmp_path, state_hash, grads, chains):
    """Silence is not "off". A term the descriptor does not state is
    required; only an explicit false drops its member."""
    from gensyn_audit import handoff as handoff_mod

    d = write_handoff(tmp_path / "h", world=2)
    meta = json.loads((d / "meta.json").read_text())
    meta["config_resolved"] = json.dumps({"train": {"state_hash": state_hash}})
    (d / "meta.json").write_text(json.dumps(meta))
    h = handoff_mod.inspect(d)
    assert (h.gradients_required, h.batch_chains_required) == (grads, chains)


@pytest.mark.parametrize(
    "resolved",
    [
        json.dumps({"train": {}}),
        json.dumps({}),
    ],
)
def test_a_descriptor_without_the_block_needs_everything(tmp_path, resolved):
    from gensyn_audit import handoff as handoff_mod

    d = write_handoff(
        tmp_path / "h", world=2, omit=("gradients.safetensors", "batch_hasher.rank_1.bin")
    )
    meta = json.loads((d / "meta.json").read_text())
    meta["config_resolved"] = resolved
    (d / "meta.json").write_text(json.dumps(meta))
    h = handoff_mod.inspect(d)
    assert h.gradients_required and h.batch_chains_required
    assert {"gradients.safetensors", "batch_hasher.rank_1.bin"} <= set(h.missing)


@pytest.mark.parametrize(
    "resolved",
    [
        {"train": {"state_hash": _OPEN_8B_HASH}},  # an object, not the JSON string
        "{not json",
        json.dumps([]),
        json.dumps({"train": []}),
        json.dumps({"train": {"state_hash": "off"}}),
        json.dumps({"train": {"state_hash": {**_OPEN_8B_HASH, "include_grads": "false"}}}),
        json.dumps({"train": {"state_hash": {**_OPEN_8B_HASH, "include_batch": 0}}}),
        json.dumps({"train": {"state_hash": {**_OPEN_8B_HASH, "include_grads": None}}}),
        json.dumps({"train": {"state_hash": {**_OPEN_8B_HASH, "every_n_steps": True}}}),
        json.dumps({"train": {"state_hash": {**_OPEN_8B_HASH, "every_n_steps": -1}}}),
        json.dumps({"train": {"state_hash": {**_OPEN_8B_HASH, "every_n_steps": "1"}}}),
    ],
)
def test_a_malformed_hash_contract_is_refused_not_relaxed(tmp_path, resolved):
    """A value that cannot be read is not taken to mean the gradients and
    chains are optional: the hand-off is refused outright."""
    from gensyn_audit.errors import AuditError
    from gensyn_audit.handoff import inspect

    d = write_handoff(tmp_path / "h", world=2, state_hash=_OPEN_8B_HASH)
    meta = json.loads((d / "meta.json").read_text())
    meta["config_resolved"] = resolved
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(AuditError, match="malformed config_resolved"):
        inspect(d)


@pytest.mark.parametrize("step", [1, 100])
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("packed", [False, True])
def test_cli_with_real_checkpoint_and_native_verifier(world, capsys, step, device, packed):
    """Only kit/record discovery is a fixture; the checkpoint and subprocess
    verifier are real. Run in an environment with both projects installed."""
    import sys

    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    pytest.importorskip("pretrain")
    pytest.importorskip("repop")
    import dataclasses

    from pretrain.cli.audit_replay import _save_chained_audit_checkpoint
    from pretrain.config import load_config, parse_config_resolved
    from pretrain.data.global_stream import GlobalStreamState
    from pretrain.model import build_model
    from pretrain.optim.adamw_repop import prime_optimizer_state
    from pretrain.optim.registry import build_optimizer
    from pretrain.parallel.parallel_dims import ParallelDims
    from pretrain.parallel.parallelize_llama3_repop import parallelize_llama3_repop
    from pretrain.train.checkpoint import CheckpointMeta
    from pretrain.train.state_hash import (
        RunningBatchHasher,
        audit_shard_state_digest,
        combine_batch_digests,
        finalize_state_hash,
    )

    actual_entry = Path(sys.executable).with_name("pretrain-audit-verify-handoff")
    assert actual_entry.is_file(), "Install the handoff verifier in this test environment"
    entry = world["venv_bin"] / actual_entry.name
    entry.unlink()
    entry.symlink_to(actual_entry)

    doc = json.loads(load_config("100m_smoke_repop").model_dump_json())
    doc["model"].update(
        n_layers=1,
        d_model=32,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_intermediate=64,
        vocab_size=128,
        max_seq_len_pretrain=16,
    )
    doc["train"]["seq_len"] = doc["data"]["seq_len"] = 8
    cfg = parse_config_resolved(json.dumps(doc))
    model = parallelize_llama3_repop(
        build_model(cfg.model, device="cpu"),
        cfg,
        ParallelDims(dp_replicate=1, dp_shard=1, world_size=1),
    )
    optimizer = build_optimizer(model, cfg.optim)
    prime_optimizer_state(optimizer)
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.125)
    optimizer.step()
    hashers = [RunningBatchHasher() for _ in range(4)]
    for r, h in enumerate(hashers):
        h.update({"input_ids": torch.full((1, 4), r, dtype=torch.int64)})
    batch = combine_batch_digests([h.local_digest() for h in hashers])
    prev = "0" * 64 if step == 1 else "deadbeef" * 8
    published = finalize_state_hash(
        prev_hash=prev,
        optimizer=optimizer,
        batch_digest=batch,
        shard_state_digest=audit_shard_state_digest(
            model, 2, 4, optimizer=optimizer, include_grads=True
        ),
    )
    held = {n: p.grad.detach().cpu() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    meta = CheckpointMeta(
        step=step,
        consumed_tokens=step * 32,
        git_sha="test",
        config_resolved=cfg.model_dump_json(),
        tokenizer_hash="test",
        container_digest="test",
        chained_hash=published,
        reduction_mode="deterministic_allgather",
        dp_world_size=4,
        dp_shard=2,
        dp_replicate=2,
        repop_env={"REPOP_EXECUTION_MODE": "cross_device_reproducible"},
    )
    saved = _save_chained_audit_checkpoint(
        save_dir=str(world["tmp"] / "real-handoff"),
        step=step,
        consumed=step * 32,
        digest=published,
        chained_hash_meta=published,
        meta_obj=dataclasses.asdict(meta),
        model=model,
        optimizer=optimizer,
        stream_state=GlobalStreamState(
            consumed_documents_per_source={"web": 1}, epoch_per_source={"web": 0}, windows_emitted=1
        ),
        spike_state={},
        batch_hashers=hashers,
        batch_digest=batch,
        N=4,
        gradients=held,
    )
    hashes = {0: "f" * 64, step: published, step + 1: "2" * 64}
    if step > 1:
        hashes[step - 1] = prev
    world["log"].write_text(
        "".join(json.dumps({"step": s, "state_hash": h}) + "\n" for s, h in sorted(hashes.items()))
    )
    artifact = saved
    artifact_root = saved
    if packed:
        from types import SimpleNamespace

        from gensyn_audit import convert, upload
        from gensyn_audit.outcome import Outcome
        from gensyn_audit.plan import Workdir
        from gensyn_audit.record import UploadTicket

        converter = Path(sys.executable).with_name(convert.ENTRYPOINT)
        assert converter.is_file(), "install the current converter in this environment"
        (world["venv_bin"] / converter.name).symlink_to(converter)
        artifact = cli._pack_handoff(
            SimpleNamespace(
                venv=world["venv_bin"].parent, workdir=Workdir(world["tmp"] / "sender")
            ),
            SimpleNamespace(unit_kind="interval"),
            SimpleNamespace(saved_checkpoint=str(saved)),
            Outcome.MATCH,
        )
        assert artifact is not None
        bundle = upload.build_bundle(artifact, world["tmp"])
        assert bundle.path == artifact and bundle.digest == upload.digest_file(artifact)
        delivered = world["tmp"] / "delivered" / artifact.name
        upload.send(bundle, UploadTicket(signed_url=delivered.as_uri()))
        assert upload.digest_file(delivered) == bundle.digest
        artifact = delivered
        artifact_root = delivered.parent
    pred = {
        "source": "crowd-provided checkpoint",
        "step": step,
        "uri": str(artifact),
        "digest": upload.digest_file(artifact) if packed else published,
        **({} if packed else {"artifactFiles": _artifact_files(artifact_root)}),
    }
    if packed:
        pred.pop("step")
    record = {
        "run": RUN,
        "steps": {str(step): {"committed_hash": hashes[step + 1], "predecessor": pred}},
    }
    fixture = world["tmp"] / "real-record.json"
    fixture.write_text(json.dumps(record))
    args = [
        "verify",
        "--kit",
        str(world["kit"]),
        "--step",
        str(step),
        "--run",
        RUN,
        "--record",
        f"mock://{fixture}",
        "--state-hashes",
        str(world["log"]),
        "--workdir",
        str(world["tmp"] / "receiver"),
        "--device",
        device,
        "--no-color",
    ]
    assert cli.main(args) == 0
    assert "tensor state reconstructed and matched" in capsys.readouterr().out

    from safetensors import safe_open
    from safetensors.torch import save_file

    if packed:
        restored = world["tmp"] / "receiver" / "verified-predecessor"
        for name in (
            "meta.json",
            "state_hash.txt",
            "global_stream.json",
            *[f"batch_hasher.rank_{r}.bin" for r in range(4)],
        ):
            assert (restored / name).read_bytes() == (saved / name).read_bytes()
        with (
            safe_open(str(restored / "gradients.safetensors"), framework="pt") as received,
            safe_open(str(saved / "gradients.safetensors"), framework="pt") as sent,
        ):
            names = sent.keys()
            assert received.metadata() == sent.metadata()
            assert received.keys() == names
            assert all(torch.equal(received.get_tensor(n), sent.get_tensor(n)) for n in names)
        # A corrupted unpack cache must not be reused just because it exists.
        (restored / "gradients.safetensors").write_bytes(b"corrupt cache")
        assert cli.main(args) == 0
        capsys.readouterr()

    sidecar = saved / "gradients.safetensors"
    with safe_open(str(sidecar), framework="pt") as f:
        names = f.keys()
        gradients = {n: f.get_tensor(n) for n in names}
        metadata = f.metadata()
    gradients[min(gradients)].view(-1)[0] += 1
    save_file(gradients, str(sidecar), metadata=metadata)
    if packed:
        convert.pack(world["venv_bin"].parent, saved, artifact)
    # A byte edit first fails the artifact gate, without opening the checkpoint.
    assert cli.main(args) == cli.EXIT_ERROR
    assert "not the file the record published" in capsys.readouterr().err
    # Even if these new bytes are catalogued, they must still match the ORIGINAL
    # published training commitment. Exercise the actual verifier mismatch path.
    if packed:
        pred["digest"] = upload.digest_file(artifact)
    else:
        pred["artifactFiles"] = _artifact_files(artifact_root)
    fixture.write_text(json.dumps(record))
    assert cli.main(args) == cli.EXIT_ERROR
    assert "verifier failed" in capsys.readouterr().err


def test_verify_reports_a_checkpoint_not_downloaded_yet(world, capsys):
    import shutil

    pred = _anchor(world)
    fixture = _fixture(world, pred, kind="published")
    shutil.rmtree(pred)
    assert _verify(world, fixture) == cli.EXIT_ERROR
    assert "Nothing downloaded here yet" in capsys.readouterr().err


@pytest.mark.parametrize("size", ["abc", None, {}, -1, float("inf")])
def test_invalid_manifest_sizes_are_dropped(size):
    from gensyn_audit.record import parse_artifact_files

    assert parse_artifact_files([{"name": "meta.json", "sha256": "a" * 64, "bytes": size}]) == ()
    assert parse_artifact_files(size) == ()


def test_verify_defaults_to_cpu():
    assert cli.build_parser().parse_args(["verify"]).device == "cpu"
    assert cli.build_parser().parse_args(["verify", "--device", "mps"]).device == "mps"


def test_crowd_missing_record_step_uses_requested_audit_step(world, capsys):
    fixture = _fixture(world, _crowd(world), kind="crowd-provided checkpoint")
    doc = json.loads(fixture.read_text())
    doc["steps"]["100"]["predecessor"].pop("step")
    fixture.write_text(json.dumps(doc))
    assert _verify(world, fixture) == cli.EXIT_OK
    assert "tensor state reconstructed and matched" in capsys.readouterr().out
