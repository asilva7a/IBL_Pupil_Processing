"""Stage 1: discover/load behavioral sessions and build the canonical trial table.

The script supports two execution modes:

1. Remote ONE mode (default): discover QC-passing IBL ChoiceWorld sessions.
2. Local-table mode (``--input-table``): canonicalize an existing CSV/Parquet
   export without contacting ONE.  This mode is useful for testing and for
   rebuilding downstream stages from an archived behavioral table.

Credentials are never embedded here.  Remote mode uses the user's existing ONE
configuration and the base URL in :mod:`config`.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import config
from utils import (
    build_signed_contrast,
    choice_trial_mask,
    configure_stage_logger,
    encode_ibl_choice,
    encode_reward,
    ensure_directory,
    load_table,
    require_columns,
    require_unique_key,
    save_table,
    validate_trial_order,
)

LOG_PATH = config.LOG_DIR / "01_build_trial_table.log"


def _format_duration(seconds: float) -> str:
    """Format an elapsed duration for progress messages."""

    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _progress(
    logger: Any | None,
    message: str,
    *args: Any,
    enabled: bool = True,
) -> None:
    """Emit a flushed console/file progress message through the stage logger."""

    if enabled and logger is not None:
        logger.info(message, *args)


def label_trial_epochs(
    trials: pd.DataFrame,
    *,
    probability_left_column: str = config.BLOCK_PRIOR_COLUMN,
    n_transition_trials: int = config.TRANSITION_EPOCH_TRIALS,
) -> pd.DataFrame:
    """Label unbiased, transition, and stable epochs within one session.

    The implementation reproduces the notebook rule: trials with
    ``probabilityLeft == 0.5`` are unbiased; after each change into a biased
    block, the next ``n_transition_trials`` stable trials are relabeled as
    transition.
    """

    if n_transition_trials < 0:
        raise ValueError("n_transition_trials must be nonnegative.")
    require_columns(trials, [probability_left_column], table_name="session trials")

    result = trials.copy().reset_index(drop=True)
    probability_left = pd.to_numeric(
        result[probability_left_column], errors="coerce"
    ).to_numpy(float)
    epoch = np.full(len(result), "stable", dtype=object)
    epoch[np.isclose(probability_left, 0.5, equal_nan=False)] = "unbiased"

    changed = np.zeros(len(result), dtype=bool)
    if len(result) > 1:
        changed[1:] = ~np.isclose(
            probability_left[1:], probability_left[:-1], equal_nan=True
        )

    for transition_index in np.flatnonzero(changed):
        if np.isclose(probability_left[transition_index], 0.5, equal_nan=False):
            continue
        stop = min(transition_index + n_transition_trials, len(result))
        mask = epoch[transition_index:stop] == "stable"
        epoch[transition_index:stop][mask] = "transition"

    result[config.EPOCH_COLUMN] = epoch
    return result


def _create_one() -> Any:
    """Create a ONE client using the local authentication configuration."""

    os.environ.setdefault(
        "ONE_HTTP_DL_THREADS", str(config.ONE_HTTP_DOWNLOAD_THREADS)
    )
    try:
        from one.api import ONE
    except ImportError as error:  # pragma: no cover - depends on external package
        raise RuntimeError(
            "Remote mode requires ONE-api. Install the project requirements or "
            "run with --input-table."
        ) from error

    # Different ONE releases accept slightly different constructor keywords.
    constructor_attempts = (
        {"base_url": config.ONE_BASE_URL, "mode": config.ONE_QUERY_TYPE},
        {"base_url": config.ONE_BASE_URL},
        {},
    )
    last_error: Exception | None = None
    for kwargs in constructor_attempts:
        try:
            return ONE(**kwargs)
        except Exception as error:  # pragma: no cover - external configuration
            last_error = error
    raise RuntimeError("Could not initialize ONE from the local configuration.") from last_error


def is_choiceworld(one: Any, eid: str) -> bool:
    """Return whether a session is a behavioral ChoiceWorld protocol."""

    try:
        protocol = one.get_details(str(eid)).get("task_protocol", "") or ""
    except Exception:
        return False
    normalized = str(protocol).lower()
    excluded = ("passive", "replay", "spontaneous", "habituation")
    return "choiceworld" in normalized and not any(x in normalized for x in excluded)


def find_video_sessions(
    one: Any,
    subject: str,
    *,
    logger: Any | None = None,
    show_progress: bool = True,
) -> list[str]:
    """Find subject sessions with a raw left-camera movie."""

    _progress(
        logger,
        "Subject %s: querying raw left-camera video sessions",
        subject,
        enabled=show_progress,
    )
    try:
        eids = one.search(
            subject=subject,
            datasets=["_iblrig_leftCamera.raw.mp4"],
            query_type=config.ONE_QUERY_TYPE,
        )
    except Exception as error:
        _progress(
            logger,
            "Subject %s: video-session query failed: %s",
            subject,
            error,
            enabled=show_progress,
        )
        return []
    result = [str(eid) for eid in eids]
    _progress(
        logger,
        "Subject %s: found %d raw-video sessions",
        subject,
        len(result),
        enabled=show_progress,
    )
    return result


def find_pupil_sessions(
    one: Any,
    subject: str,
    *,
    logger: Any | None = None,
    show_progress: bool = True,
    progress_every: int = 10,
) -> list[str]:
    """Find ChoiceWorld sessions with precomputed left-camera DLC."""

    _progress(
        logger,
        "Subject %s: querying left-camera DLC sessions",
        subject,
        enabled=show_progress,
    )
    try:
        eids = one.search(
            subject=subject,
            datasets=["_ibl_leftCamera.dlc.pqt"],
            query_type=config.ONE_QUERY_TYPE,
        )
    except Exception as error:
        _progress(
            logger,
            "Subject %s: DLC-session query failed: %s",
            subject,
            error,
            enabled=show_progress,
        )
        return []

    eids = [str(eid) for eid in eids]
    _progress(
        logger,
        "Subject %s: checking task protocol for %d DLC sessions",
        subject,
        len(eids),
        enabled=show_progress,
    )
    choiceworld_eids: list[str] = []
    for index, eid in enumerate(eids, start=1):
        if is_choiceworld(one, eid):
            choiceworld_eids.append(eid)
        if progress_every > 0 and (index % progress_every == 0 or index == len(eids)):
            _progress(
                logger,
                "Subject %s: protocol checks %d/%d (%d ChoiceWorld)",
                subject,
                index,
                len(eids),
                len(choiceworld_eids),
                enabled=show_progress,
            )
    return choiceworld_eids


def get_sex(one: Any, subject: str) -> str | None:
    """Read the subject's recorded sex from Alyx."""

    try:
        return one.alyx.rest("subjects", "read", id=subject).get("sex")
    except Exception:
        return None


def load_session_trials(one: Any, eid: str) -> pd.DataFrame:
    """Load one session's trial table with a SessionLoader-compatible fallback."""

    try:
        from brainbox.io.one import SessionLoader
    except ImportError:
        SessionLoader = None  # type: ignore[assignment]

    if SessionLoader is not None:
        session_loader = SessionLoader(eid=eid, one=one)
        session_loader.load_trials()
        trials = session_loader.trials.copy()
    else:  # pragma: no cover - external ONE execution path
        try:
            loaded = one.load_object(eid, "trials")
        except Exception as error:
            raise RuntimeError(
                "Loading trials requires brainbox/ibllib or ONE.load_object support."
            ) from error
        trials = pd.DataFrame(loaded)

    return trials.reset_index(drop=True)


def canonicalize_session_trials(
    trials: pd.DataFrame,
    *,
    subject: str,
    eid: str,
    sequence_id: int,
    sex: str | None,
) -> pd.DataFrame:
    """Add canonical keys, encodings, and epoch labels to one session."""

    required = [
        config.CHOICE_COLUMN,
        config.FEEDBACK_COLUMN,
        config.CONTRAST_LEFT_COLUMN,
        config.CONTRAST_RIGHT_COLUMN,
        config.BLOCK_PRIOR_COLUMN,
    ]
    require_columns(trials, required, table_name=f"trials for {eid}")

    result = trials.copy().reset_index(drop=True)
    result[config.SUBJECT_COLUMN] = str(subject)
    result[config.SEX_COLUMN] = sex
    result[config.SESSION_COLUMN] = str(eid)
    result[config.SEQUENCE_COLUMN] = int(sequence_id)
    result[config.TRIAL_INDEX_COLUMN] = np.arange(len(result), dtype=int)

    result[config.SIGNED_CONTRAST_COLUMN] = build_signed_contrast(result)
    result[config.REWARD_COLUMN] = encode_reward(result[config.FEEDBACK_COLUMN])
    result[config.ACCURACY_COLUMN] = result[config.REWARD_COLUMN]
    result["rightward_choice"] = encode_ibl_choice(result[config.CHOICE_COLUMN])
    result["choice_valid"] = choice_trial_mask(result[config.CHOICE_COLUMN])
    result = label_trial_epochs(result)
    return result


def _subject_sequence_map(table: pd.DataFrame) -> pd.DataFrame:
    """Assign stable, deterministic sequence IDs from session IDs."""

    sessions = table[[config.SUBJECT_COLUMN, config.SESSION_COLUMN]].drop_duplicates()
    sessions = sessions.sort_values(
        [config.SUBJECT_COLUMN, config.SESSION_COLUMN], kind="stable"
    )
    sessions[config.SEQUENCE_COLUMN] = sessions.groupby(
        config.SUBJECT_COLUMN, sort=False
    ).cumcount()
    return sessions


def canonicalize_local_table(
    table: pd.DataFrame,
    *,
    logger: Any | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Canonicalize an existing multi-session table without contacting ONE."""

    require_columns(
        table,
        [
            config.SUBJECT_COLUMN,
            config.SESSION_COLUMN,
            config.CHOICE_COLUMN,
            config.FEEDBACK_COLUMN,
            config.CONTRAST_LEFT_COLUMN,
            config.CONTRAST_RIGHT_COLUMN,
            config.BLOCK_PRIOR_COLUMN,
        ],
        table_name="local input table",
    )

    working = table.copy()
    working[config.SUBJECT_COLUMN] = working[config.SUBJECT_COLUMN].astype(str)
    working[config.SESSION_COLUMN] = working[config.SESSION_COLUMN].astype(str)

    if config.BEHAVIOR_SUBJECT_EXCLUSIONS:
        working = working.loc[
            ~working[config.SUBJECT_COLUMN].isin(config.BEHAVIOR_SUBJECT_EXCLUSIONS)
        ].copy()
    if config.SUBJECT_ALLOWLIST is not None:
        working = working.loc[
            working[config.SUBJECT_COLUMN].isin(config.SUBJECT_ALLOWLIST)
        ].copy()

    if config.SEQUENCE_COLUMN not in working:
        mapping = _subject_sequence_map(working)
        working = working.merge(
            mapping,
            on=[config.SUBJECT_COLUMN, config.SESSION_COLUMN],
            how="left",
            validate="many_to_one",
        )

    if config.TRIAL_INDEX_COLUMN not in working:
        order_candidates = [
            config.STIMULUS_TIME_COLUMN,
            "intervals_0",
            "goCue_times",
        ]
        order_column = next((c for c in order_candidates if c in working), None)
        if order_column is not None:
            working = working.sort_values(
                [config.SUBJECT_COLUMN, config.SESSION_COLUMN, order_column],
                kind="stable",
            )
        working[config.TRIAL_INDEX_COLUMN] = working.groupby(
            [config.SUBJECT_COLUMN, config.SESSION_COLUMN], sort=False
        ).cumcount()

    sex_lookup = (
        working[[config.SUBJECT_COLUMN, config.SEX_COLUMN]]
        .drop_duplicates(config.SUBJECT_COLUMN)
        .set_index(config.SUBJECT_COLUMN)[config.SEX_COLUMN]
        .to_dict()
        if config.SEX_COLUMN in working
        else {}
    )

    pieces = []
    manifest_rows = []
    grouped_sessions = list(
        working.groupby([config.SUBJECT_COLUMN, config.SESSION_COLUMN], sort=True)
    )
    started = time.perf_counter()
    _progress(
        logger,
        "Local mode: canonicalizing %d sessions",
        len(grouped_sessions),
        enabled=show_progress,
    )
    for session_number, ((subject, eid), session) in enumerate(
        grouped_sessions, start=1
    ):
        _progress(
            logger,
            "Local session %d/%d | subject=%s | eid=%s | rows=%d",
            session_number,
            len(grouped_sessions),
            subject,
            eid,
            len(session),
            enabled=show_progress,
        )
        sequence_values = session[config.SEQUENCE_COLUMN].dropna().unique()
        if len(sequence_values) != 1:
            raise ValueError(f"Session {eid} has multiple sequence IDs: {sequence_values}")
        session = session.sort_values(config.TRIAL_INDEX_COLUMN, kind="stable")
        canonical = canonicalize_session_trials(
            session,
            subject=str(subject),
            eid=str(eid),
            sequence_id=int(sequence_values[0]),
            sex=sex_lookup.get(str(subject)),
        )
        pieces.append(canonical)
        manifest_rows.append(
            {
                config.SUBJECT_COLUMN: str(subject),
                config.SESSION_COLUMN: str(eid),
                config.SEQUENCE_COLUMN: int(sequence_values[0]),
                "load_status": "local",
                "n_trials": len(canonical),
                "n_choice_trials": int(canonical["choice_valid"].sum()),
            }
        )
        elapsed = time.perf_counter() - started
        _progress(
            logger,
            "Local progress %d/%d | cumulative trials=%d | elapsed=%s",
            session_number,
            len(grouped_sessions),
            sum(len(piece) for piece in pieces),
            _format_duration(elapsed),
            enabled=show_progress,
        )

    trial_table = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    qc = build_subject_qc(manifest, trial_table)
    candidates = qc.loc[qc["passes_qc"], config.SUBJECT_COLUMN]
    trial_table = trial_table.loc[
        trial_table[config.SUBJECT_COLUMN].isin(candidates)
    ].copy()
    manifest["included"] = manifest[config.SUBJECT_COLUMN].isin(candidates)
    return trial_table, qc, manifest


def build_subject_qc(manifest: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Create the subject-level availability and trial-count QC table."""

    if manifest.empty:
        return pd.DataFrame(
            columns=[
                config.SUBJECT_COLUMN,
                config.SEX_COLUMN,
                "n_video",
                "n_dlc",
                "n_trials_total",
                "n_choice_total",
                "passes_qc",
            ]
        )

    trial_counts = (
        trials.groupby(config.SUBJECT_COLUMN, sort=False)
        .agg(
            n_trials_total=(config.TRIAL_INDEX_COLUMN, "size"),
            n_choice_total=("choice_valid", "sum"),
        )
        .reset_index()
    )
    manifest_counts = (
        manifest.groupby(config.SUBJECT_COLUMN, sort=False)
        .agg(
            n_dlc=(config.SESSION_COLUMN, "nunique"),
            n_video=("has_video", "sum") if "has_video" in manifest else (config.SESSION_COLUMN, "nunique"),
        )
        .reset_index()
    )
    qc = manifest_counts.merge(
        trial_counts, on=config.SUBJECT_COLUMN, how="outer", validate="one_to_one"
    )
    if config.SEX_COLUMN in trials:
        sex = (
            trials[[config.SUBJECT_COLUMN, config.SEX_COLUMN]]
            .drop_duplicates(config.SUBJECT_COLUMN)
        )
        qc = qc.merge(sex, on=config.SUBJECT_COLUMN, how="left", validate="one_to_one")
    else:
        qc[config.SEX_COLUMN] = None

    trial_metric = "n_choice_total" if config.USE_CHOICE_TRIALS_FOR_QC else "n_trials_total"
    qc["passes_qc"] = (
        qc["n_dlc"].fillna(0).ge(config.MIN_DLC_SESSIONS)
        & qc[trial_metric].fillna(0).ge(config.MIN_TOTAL_TRIALS)
    )
    qc["trial_qc_metric"] = trial_metric
    return qc.sort_values(config.SUBJECT_COLUMN).reset_index(drop=True)


def discover_remote_table(
    one: Any,
    logger: Any,
    *,
    subjects_override: Iterable[str] | None = None,
    show_progress: bool = True,
    progress_every: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discover remote sessions, load trials, and apply subject-level QC."""

    stage_started = time.perf_counter()
    if subjects_override is not None:
        subjects = sorted({str(subject) for subject in subjects_override})
        _progress(
            logger,
            "Using command-line subject list (%d subjects)",
            len(subjects),
            enabled=show_progress,
        )
    elif config.SUBJECT_ALLOWLIST is not None:
        subjects = sorted(map(str, config.SUBJECT_ALLOWLIST))
        _progress(
            logger,
            "Using configured subject allowlist (%d subjects)",
            len(subjects),
            enabled=show_progress,
        )
    else:
        _progress(
            logger,
            "Querying ONE for all sessions containing _ibl_leftCamera.dlc.pqt",
            enabled=show_progress,
        )
        try:
            dlc_eids = list(
                one.search(
                    datasets=["_ibl_leftCamera.dlc.pqt"],
                    query_type=config.ONE_QUERY_TYPE,
                )
            )
        except Exception as error:  # pragma: no cover - remote
            raise RuntimeError("ONE dataset discovery failed.") from error

        _progress(
            logger,
            "ONE returned %d DLC sessions; resolving subject identities",
            len(dlc_eids),
            enabled=show_progress,
        )
        subject_set: set[str] = set()
        for index, eid in enumerate(dlc_eids, start=1):
            try:
                details = one.get_details(str(eid))
            except Exception as error:  # pragma: no cover - remote
                logger.warning("Could not resolve session %s: %s", eid, error)
                details = {}
            subject = details.get("subject")
            if subject:
                subject_set.add(str(subject))
            if progress_every > 0 and (index % progress_every == 0 or index == len(dlc_eids)):
                _progress(
                    logger,
                    "Subject discovery %d/%d sessions | %d unique subjects | elapsed=%s",
                    index,
                    len(dlc_eids),
                    len(subject_set),
                    _format_duration(time.perf_counter() - stage_started),
                    enabled=show_progress,
                )
        subjects = sorted(subject_set)

    _progress(
        logger,
        "Discovered %d candidate subjects",
        len(subjects),
        enabled=show_progress,
    )
    session_records: list[dict[str, Any]] = []
    all_trials: list[pd.DataFrame] = []
    total_sessions_seen = 0
    total_sessions_loaded = 0
    cumulative_trials = 0

    for subject_number, subject in enumerate(subjects, start=1):
        if subject in config.BEHAVIOR_SUBJECT_EXCLUSIONS:
            _progress(
                logger,
                "Subject %d/%d | %s | skipped by pre-existing exclusion",
                subject_number,
                len(subjects),
                subject,
                enabled=show_progress,
            )
            continue

        subject_started = time.perf_counter()
        _progress(
            logger,
            "Subject %d/%d | %s | starting discovery",
            subject_number,
            len(subjects),
            subject,
            enabled=show_progress,
        )
        sex = get_sex(one, subject)
        pupil_sessions = sorted(
            find_pupil_sessions(
                one,
                subject,
                logger=logger,
                show_progress=show_progress,
                progress_every=progress_every,
            )
        )
        video_sessions = set(
            find_video_sessions(
                one,
                subject,
                logger=logger,
                show_progress=show_progress,
            )
        )
        _progress(
            logger,
            "Subject %d/%d | %s | %d ChoiceWorld DLC sessions; %d video sessions",
            subject_number,
            len(subjects),
            subject,
            len(pupil_sessions),
            len(video_sessions),
            enabled=show_progress,
        )

        for sequence_id, eid in enumerate(pupil_sessions):
            total_sessions_seen += 1
            session_started = time.perf_counter()
            _progress(
                logger,
                "Subject %d/%d | %s | session %d/%d | loading %s",
                subject_number,
                len(subjects),
                subject,
                sequence_id + 1,
                len(pupil_sessions),
                eid,
                enabled=show_progress,
            )
            record: dict[str, Any] = {
                config.SUBJECT_COLUMN: subject,
                config.SEX_COLUMN: sex,
                config.SESSION_COLUMN: eid,
                config.SEQUENCE_COLUMN: sequence_id,
                "has_video": eid in video_sessions,
                "load_status": "pending",
                "load_error": "",
                "n_trials": 0,
                "n_choice_trials": 0,
            }
            try:
                raw = load_session_trials(one, eid)
                canonical = canonicalize_session_trials(
                    raw,
                    subject=subject,
                    eid=eid,
                    sequence_id=sequence_id,
                    sex=sex,
                )
                all_trials.append(canonical)
                n_trials = len(canonical)
                n_choice = int(canonical["choice_valid"].sum())
                cumulative_trials += n_trials
                total_sessions_loaded += 1
                record.update(
                    load_status="success",
                    n_trials=n_trials,
                    n_choice_trials=n_choice,
                )
                _progress(
                    logger,
                    "Loaded %s | trials=%d | choices=%d | session time=%s | "
                    "cumulative sessions=%d/%d | cumulative trials=%d",
                    eid,
                    n_trials,
                    n_choice,
                    _format_duration(time.perf_counter() - session_started),
                    total_sessions_loaded,
                    total_sessions_seen,
                    cumulative_trials,
                    enabled=show_progress,
                )
            except Exception as error:  # pragma: no cover - remote
                record.update(
                    load_status="failed",
                    load_error=f"{type(error).__name__}: {error}",
                )
                logger.warning("Failed %s %s: %s", subject, eid, error)
            session_records.append(record)

        _progress(
            logger,
            "Subject %d/%d | %s complete | elapsed=%s | total loaded sessions=%d | "
            "total trials=%d",
            subject_number,
            len(subjects),
            subject,
            _format_duration(time.perf_counter() - subject_started),
            total_sessions_loaded,
            cumulative_trials,
            enabled=show_progress,
        )

    manifest = pd.DataFrame(session_records)
    trial_table = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
    if trial_table.empty:
        raise RuntimeError("No behavioral sessions were loaded successfully.")

    _progress(
        logger,
        "All session loads finished | successful=%d/%d | raw trials=%d | elapsed=%s",
        total_sessions_loaded,
        total_sessions_seen,
        len(trial_table),
        _format_duration(time.perf_counter() - stage_started),
        enabled=show_progress,
    )
    qc = build_subject_qc(manifest.loc[manifest["load_status"] == "success"], trial_table)
    candidates = set(qc.loc[qc["passes_qc"], config.SUBJECT_COLUMN])
    trial_table = trial_table.loc[
        trial_table[config.SUBJECT_COLUMN].isin(candidates)
    ].copy()
    manifest["included"] = manifest[config.SUBJECT_COLUMN].isin(candidates)
    _progress(
        logger,
        "Subject QC retained %d/%d subjects and %d trials",
        len(candidates),
        qc[config.SUBJECT_COLUMN].nunique(),
        len(trial_table),
        enabled=show_progress,
    )
    return trial_table, qc, manifest


def validate_trial_table(table: pd.DataFrame) -> None:
    """Run the stage acceptance checks that can be automated."""

    require_unique_key(table, config.TRIAL_KEY_COLUMNS, table_name="trial table")
    validate_trial_order(
        table,
        [config.SUBJECT_COLUMN, config.SESSION_COLUMN],
        config.TRIAL_INDEX_COLUMN,
        table_name="trial table",
    )
    choice = encode_ibl_choice(pd.Series([config.IBL_RIGHT_CHOICE])).iloc[0]
    if choice != config.ENCODED_RIGHT_CHOICE:
        raise AssertionError("IBL -1 did not map to a rightward choice.")

    easy = table.loc[table[config.SIGNED_CONTRAST_COLUMN].abs().ge(0.5)]
    if len(easy) >= 20:
        right_easy = easy.loc[easy[config.SIGNED_CONTRAST_COLUMN] > 0, "rightward_choice"].mean()
        left_easy = easy.loc[easy[config.SIGNED_CONTRAST_COLUMN] < 0, "rightward_choice"].mean()
        if np.isfinite(right_easy) and np.isfinite(left_easy) and right_easy <= left_easy:
            raise AssertionError(
                "Easy-trial sanity check failed: positive signed contrast does not "
                "produce more rightward choices than negative signed contrast."
            )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-table",
        type=Path,
        default=None,
        help="Canonicalize a local CSV/Parquet table instead of querying ONE.",
    )
    parser.add_argument("--output", type=Path, default=config.TRIAL_TABLE_PATH)
    parser.add_argument("--subject-qc", type=Path, default=config.SUBJECT_QC_PATH)
    parser.add_argument(
        "--session-manifest", type=Path, default=config.SESSION_MANIFEST_PATH
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help=(
            "Optional subject IDs for a small remote smoke test. This overrides "
            "SUBJECT_ALLOWLIST for the current run."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help=(
            "During remote discovery, report protocol/subject-resolution progress "
            "every N items (default: 10)."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress detailed progress messages; warnings and final summary remain.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config.ensure_project_directories()
    logger = configure_stage_logger("01_build_trial_table", log_path=LOG_PATH)

    if args.input_table is not None:
        logger.info("Using local input table: %s", args.input_table)
        trial_table, qc, manifest = canonicalize_local_table(
            load_table(args.input_table),
            logger=logger,
            show_progress=not args.no_progress,
        )
    else:
        logger.info("Using remote ONE discovery at %s", config.ONE_BASE_URL)
        trial_table, qc, manifest = discover_remote_table(
            _create_one(),
            logger,
            subjects_override=args.subjects,
            show_progress=not args.no_progress,
            progress_every=max(1, args.progress_every),
        )

    trial_table = trial_table.sort_values(
        list(config.TRIAL_KEY_COLUMNS), kind="stable"
    ).reset_index(drop=True)
    validate_trial_table(trial_table)

    save_table(trial_table, args.output)
    save_table(qc, args.subject_qc)
    save_table(manifest, args.session_manifest)

    summary = {
        "subjects": trial_table[config.SUBJECT_COLUMN].nunique(),
        "sessions": trial_table[config.SESSION_COLUMN].nunique(),
        "trials": len(trial_table),
        "choice_trials": int(trial_table["choice_valid"].sum()),
    }
    logger.info("Stage 1 complete: %s", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
