"""What a complete hand-off is, in one place.

A hand-off is not "a checkpoint". It is a checkpoint *plus* what the next
auditor needs and a resume does not: the target step's gradients
(`gradients.safetensors`), every DP rank's batch-hasher chain, and the
descriptor that says which of those the run hashed and how many there should
be. Without them the recipient can load the state and replay from it, but
cannot reconstruct the hash the run published for it — which is the difference
between continuing a chain and continuing someone's word for it.

So "it downloaded" and "it is usable" have to be the same question, asked in
one place: the uploader must not send a partial hand-off, the downloader must
not treat one as verified, and the preflight must size disk for all of it.

The member list mirrors what ``audit_replay --save-checkpoint-dir`` writes
(``_save_chained_audit_checkpoint``) and what it reads back on the next
interval. `spike_protocol.json` and `sampler.rank_*.json` are deliberately not
required: the writer omits them for an auditable run with no spike state, and
the reader tolerates their absence.

The gradients and the rank chains are required only when the run's state hash
folds them in, decided the way the writer decides it, from
``train.state_hash`` in the descriptor's ``config_resolved``: gradients when
``every_n_steps > 0 and include_grads``, chains when ``every_n_steps > 0 and
include_batch``. open-1b hashes both; open-8b hashes neither, and its writer
exports neither. A descriptor that does not say is held to the full contract,
and one that says something malformed is refused, so a missing field can never
shrink what has to travel. Taking this from the artifact's own meta.json is
safe for the same reason the kit's verifier gives: these are inputs to the
digest, so a hand-off that turns them off to travel light reconstructs a
different hash and fails the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import AuditError

#: The final gradients exported with the checkpoint.
GRADIENTS = "gradients.safetensors"

#: Written last by the checkpoint writer. Its absence means the writer never
#: finished or the transfer did not — never "an older format".
COMPLETE = "_COMPLETE"

#: Present in every hand-off regardless of topology or hash contract.
_ALWAYS = (
    "meta.json",
    COMPLETE,
    "state_hash.txt",
    "global_stream.json",
    "rng.rank_0.pt",
)

#: Upper bound on a declared dp_world_size. `meta.json` is the artifact
#: author's, so it decides how many files we go looking for; mirrors
#: audit_replay's own `_MAX_AUDIT_WORLD`.
MAX_WORLD = 48


@dataclass(frozen=True)
class Handoff:
    """One hand-off directory, and whether it is all there."""

    root: Path
    step: int | None
    dp_world_size: int
    present: tuple[str, ...]
    missing: tuple[str, ...]
    #: Whether the run's state hash folds in the gradients / the per-rank
    #: batch chains, so that a complete hand-off has to carry them.
    gradients_required: bool = True
    batch_chains_required: bool = True

    @property
    def is_complete(self) -> bool:
        return not self.missing

    @property
    def files(self) -> tuple[Path, ...]:
        """Every file in the hand-off, sorted — what must actually travel."""
        return tuple(sorted(p for p in self.root.rglob("*") if p.is_file()))

    @property
    def has_gradients(self) -> bool:
        return (self.root / GRADIENTS).is_file()


def _read_meta(root: Path) -> dict | None:
    """meta.json as an object, or None when it is absent.

    An unreadable meta is an error, not a default: it decides both how many
    rank chains to look for and whether to look for them at all, so the file
    that decides what to look for cannot be the one we shrug at.
    """
    import json

    if not (root / "meta.json").is_file():
        return None  # reported as a missing member by the caller
    try:
        meta = json.loads((root / "meta.json").read_text())
    except (OSError, ValueError) as exc:
        raise AuditError(
            f"{root}/meta.json is not readable JSON: {exc}",
            hint="It names the topology, which decides how many per-rank "
            "batch-hasher chains a complete hand-off has. Without it a "
            "32-rank hand-off carrying one chain looks complete.",
        ) from exc
    if not isinstance(meta, dict):
        raise AuditError(f"{root}/meta.json must contain an object.")
    return meta


def _descriptor(root: Path, meta: dict | None) -> tuple[int | None, int]:
    """`(step, dp_world_size)` from meta.json.

    Falling back to one rank would let a hand-off from a 32-rank run pass the
    completeness check while carrying only rank 0's batch chain.
    """
    if meta is None:
        return None, 1
    world = meta.get("dp_world_size")
    if type(world) is not int or not 1 <= world <= MAX_WORLD:
        raise AuditError(
            f"{root}/meta.json declares dp_world_size={world}; the cap is {MAX_WORLD}.",
            hint="meta.json comes from whoever produced the artifact, and this "
            "value decides how many per-rank files are read. A value out "
            "of range is corruption or an attempt at one.",
        )
    step = meta.get("step")
    if type(step) is not int or step <= 0:
        raise AuditError(f"{root}/meta.json must declare a positive integer step.")
    return step, world


def _hash_contract(root: Path, meta: dict | None) -> tuple[bool, bool]:
    """`(gradients, batch chains)`: which of the two the run's state hash folds in.

    Mirrors ``audit_replay``, which exports ``gradients.safetensors`` only when
    ``every_n_steps > 0 and include_grads`` and the per-rank chains only when
    ``every_n_steps > 0 and include_batch``. Every open-1b descriptor states all
    three, with both terms on.

    Absence keeps the full contract: no meta (reported missing anyway), no
    ``config_resolved``, or no value for a field all mean "require it". Only an
    explicit, well-formed ``false`` (or ``every_n_steps: 0``) drops a member.
    Anything else in those places is refused, because the alternative to
    refusing a value we cannot read is guessing which files are optional.
    """
    import json

    if meta is None or "config_resolved" not in meta:
        return True, True
    raw = meta["config_resolved"]

    def malformed(why: str) -> AuditError:
        return AuditError(
            f"{root}/meta.json has a malformed config_resolved: {why}.",
            hint="Its train.state_hash block decides whether the gradients and "
            "the per-rank batch chains are part of the published state hash, "
            "and so whether the hand-off has to carry them. A value that "
            "cannot be read is not taken to mean they are optional.",
        )

    if not isinstance(raw, str):
        raise malformed("expected a JSON string")
    try:
        config = json.loads(raw)
    except ValueError as exc:
        raise malformed(f"not JSON ({exc})") from exc
    if not isinstance(config, dict):
        raise malformed("expected an object")
    train = config.get("train", {})
    if not isinstance(train, dict):
        raise malformed("train is not an object")
    sh = train.get("state_hash", {})
    if not isinstance(sh, dict):
        raise malformed("train.state_hash is not an object")

    every = sh.get("every_n_steps", 1)
    if type(every) is not int or every < 0:
        raise malformed(f"train.state_hash.every_n_steps={every!r}")
    flags = []
    for key in ("include_grads", "include_batch"):
        value = sh.get(key, True)
        if type(value) is not bool:
            raise malformed(f"train.state_hash.{key}={value!r}")
        flags.append(every > 0 and value)
    return flags[0], flags[1]


def inspect(root: Path) -> Handoff:
    """Enumerate a hand-off directory against the member contract."""
    if not root.is_dir():
        raise AuditError(f"no hand-off directory at {root}.")
    meta = _read_meta(root)
    step, world = _descriptor(root, meta)
    grads, chains = _hash_contract(root, meta)
    wanted = [*_ALWAYS]
    if grads:
        wanted.append(GRADIENTS)
    if chains:
        wanted.extend(f"batch_hasher.rank_{r}.bin" for r in range(world))
    present = tuple(n for n in wanted if (root / n).is_file())
    missing = tuple(n for n in wanted if n not in present)
    if not (root / "dcp").is_dir() and not (root / "fallback.pt").is_file():
        missing = (*missing, "dcp/")
    return Handoff(
        root=root,
        step=step,
        dp_world_size=world,
        present=present,
        missing=missing,
        gradients_required=grads,
        batch_chains_required=chains,
    )


def require_complete(h: Handoff, *, what: str) -> None:
    """Refuse an incomplete hand-off, naming what is absent and what it costs."""
    if h.is_complete:
        return
    detail = {
        GRADIENTS: "the target step's final gradients — the published state "
        "hash folds them in, so the recipient cannot reconstruct it",
        COMPLETE: "the writer's completion marker — this checkpoint was never "
        "published, or the transfer stopped short",
        "dcp/": "the model and optimizer shards themselves",
    }
    lines = []
    for name in h.missing:
        why = detail.get(name) or (
            "a per-rank batch-hasher chain the run's batch digest is built from"
            if name.startswith("batch_hasher.")
            else "required hand-off metadata"
        )
        lines.append(f"  {name} — {why}")
    raise AuditError(
        f"{what} is incomplete: {len(h.missing)} required file(s) missing from {h.root}.",
        hint="\n".join(lines) + "\n"
        "A hand-off that cannot be checked against the published state "
        "hash must not be replayed from or passed on as if it had been.",
    )
