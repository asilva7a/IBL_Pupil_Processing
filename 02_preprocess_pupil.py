"""Stage 2: preprocess pupil traces and create trial-level pupil features.

This stage loads or reuses a cached cleaned pupil trace for every behavioral
session, extracts stimulus-locked tonic/phasic features, and also extracts a
feedback-locked phasic feature when feedback timestamps are available.  Trace
caches preserve the cleaned sample stream and preprocessing metadata so future
event-locked analyses do not need to rediscover raw datasets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import (
    configure_stage_logger,
    ensure_directory,
    load_table,
    mask_pupil_artifacts,
    require_columns,
    require_unique_key,
    robust_zscore,
    save_table,
)

LOG_PATH = config.LOG_DIR / "02_preprocess_pupil.log"


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds for compact operator-facing progress messages."""

    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _progress(
    logger: Any,
    enabled: bool,
    message: str,
    *args: Any,
) -> None:
    """Emit an INFO progress message when operator feedback is enabled."""

    if enabled:
        logger.info(message, *args)


class _SessionHeartbeat:
    """Periodically report that a long-running session step is still active."""

    def __init__(
        self,
        logger: Any,
        *,
        enabled: bool,
        interval_seconds: float,
        prefix: str,
    ) -> None:
        self.logger = logger
        self.enabled = bool(enabled and interval_seconds > 0)
        self.interval_seconds = float(interval_seconds)
        self.prefix = prefix
        self.started = time.monotonic()
        self._step = "starting"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, step: str) -> None:
        """Update the operation named in future heartbeat messages."""

        with self._lock:
            self._step = str(step)

    def __enter__(self) -> "_SessionHeartbeat":
        if self.enabled:
            self._thread = threading.Thread(
                target=self._run,
                name="pupil-progress-heartbeat",
                daemon=True,
            )
            self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            with self._lock:
                step = self._step
            self.logger.info(
                "%s | still working | step=%s | session elapsed=%s",
                self.prefix,
                step,
                _format_duration(time.monotonic() - self.started),
            )


def _create_one() -> Any:
    os.environ.setdefault(
        "ONE_HTTP_DL_THREADS", str(config.ONE_HTTP_DOWNLOAD_THREADS)
    )
    try:
        from one.api import ONE
    except ImportError as error:  # pragma: no cover - external package
        raise RuntimeError(
            "Pupil downloading requires ONE-api/ibllib. Existing trace caches may "
            "be used with --offline."
        ) from error
    for kwargs in (
        {"base_url": config.ONE_BASE_URL, "mode": config.ONE_QUERY_TYPE},
        {"base_url": config.ONE_BASE_URL},
        {},
    ):
        try:
            return ONE(**kwargs)
        except Exception:
            continue
    raise RuntimeError("Could not initialize ONE from the local configuration.")


def _cache_path(eid: str, cache_dir: Path) -> Path:
    safe_eid = str(eid).replace("/", "_")
    return cache_dir / f"{safe_eid}.npz"


def _select_dataset(dataset_names: list[str], pattern: str) -> str | None:
    matches = sorted(name for name in dataset_names if pattern in str(name))
    return matches[0] if matches else None


def fetch_and_clean_pupil_trace(
    one: Any,
    eid: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fetch DLC and timestamps, construct diameter, and mask artifacts."""

    notify = progress or (lambda _message: None)

    try:
        from brainbox.behavior.dlc import (
            get_pupil_diameter,
            get_smooth_pupil_diameter,
            likelihood_threshold,
        )
    except ImportError as error:  # pragma: no cover - external package
        raise RuntimeError("brainbox/ibllib pupil helpers are not installed.") from error

    camera = config.PUPIL_CAMERA
    notify("listing camera datasets")
    datasets = [
        str(name)
        for name in one.list_datasets(
            eid,
            filename=f"*{camera}Camera*",
            query_type=config.ONE_QUERY_TYPE,
        )
    ]
    notify(f"camera dataset listing complete | matches={len(datasets)}")
    dlc_name = _select_dataset(datasets, f"{camera}Camera.dlc")
    time_name = _select_dataset(datasets, f"{camera}Camera.times")
    if dlc_name is None or time_name is None:
        raise FileNotFoundError(f"Missing {camera}-camera DLC or timestamps for {eid}.")

    notify(f"loading DLC dataset {dlc_name}")
    dlc = one.load_dataset(eid, dlc_name)
    dlc_rows = len(dlc) if hasattr(dlc, "__len__") else "unknown"
    dlc_columns = len(getattr(dlc, "columns", []))
    notify(f"DLC loaded | rows={dlc_rows} | columns={dlc_columns}")
    notify(f"loading timestamp dataset {time_name}")
    sample_times = np.asarray(one.load_dataset(eid, time_name), dtype=float)
    finite_times = sample_times[np.isfinite(sample_times)]
    duration = (
        float(finite_times[-1] - finite_times[0])
        if finite_times.size >= 2
        else np.nan
    )
    duration_text = _format_duration(duration) if np.isfinite(duration) else "n/a"
    notify(
        f"timestamps loaded | samples={sample_times.size} | "
        f"duration={duration_text}"
    )
    notify("thresholding DLC likelihoods")
    thresholded = likelihood_threshold(
        dlc,
        threshold=config.PUPIL_DLC_LIKELIHOOD_THRESHOLD,
    )
    notify("estimating raw pupil diameter")
    raw_diameter = np.asarray(get_pupil_diameter(thresholded), dtype=float)
    raw_missing = (
        float(np.mean(~np.isfinite(raw_diameter)))
        if raw_diameter.size
        else np.nan
    )
    raw_missing_text = f"{raw_missing:.1%}" if np.isfinite(raw_missing) else "n/a"
    notify(
        f"raw pupil diameter ready | samples={raw_diameter.size} | "
        f"missing={raw_missing_text}"
    )
    notify("smoothing pupil diameter")
    smoothed = np.asarray(
        get_smooth_pupil_diameter(
            raw_diameter,
            camera,
            std_thresh=config.PUPIL_SMOOTH_STD_THRESHOLD,
            nan_thresh=config.PUPIL_SMOOTH_NAN_THRESHOLD,
        ),
        dtype=float,
    )
    smoothed_missing = (
        float(np.mean(~np.isfinite(smoothed))) if smoothed.size else np.nan
    )
    smoothed_missing_text = (
        f"{smoothed_missing:.1%}" if np.isfinite(smoothed_missing) else "n/a"
    )
    notify(f"smoothing complete | missing={smoothed_missing_text}")
    notify("masking blink and velocity artifacts")
    cleaned, artifact_mask = mask_pupil_artifacts(
        smoothed,
        sample_times,
        velocity_mad_threshold=config.BLINK_VELOCITY_MAD_THRESHOLD,
        padding_seconds=config.BLINK_PADDING_SECONDS,
        nominal_fps=config.NOMINAL_CAMERA_FPS,
    )

    finite = np.isfinite(cleaned)
    if finite.any():
        center = np.nanmedian(cleaned)
        mad = np.nanmedian(np.abs(cleaned[finite] - center))
        scale = 1.4826 * mad
        if np.isfinite(scale) and scale > np.finfo(float).eps:
            frame_rz = (cleaned - center) / scale
            extreme = np.abs(frame_rz) > config.FRAME_ROBUST_Z_MAX
            artifact_mask |= extreme
            cleaned[extreme] = np.nan
    nonphysical = np.isfinite(cleaned) & (cleaned <= 0)
    artifact_mask |= nonphysical
    cleaned[nonphysical] = np.nan
    cleaned_missing = (
        float(np.mean(~np.isfinite(cleaned))) if cleaned.size else np.nan
    )
    artifact_fraction = (float(artifact_mask.mean()) if artifact_mask.size else np.nan)
    cleaned_missing_text = (
        f"{cleaned_missing:.1%}" if np.isfinite(cleaned_missing) else "n/a"
    )
    artifact_text = (
        f"{artifact_fraction:.1%}" if np.isfinite(artifact_fraction) else "n/a"
    )
    notify(
        f"trace cleaning complete | cleaned missing={cleaned_missing_text} | "
        f"artifact fraction={artifact_text}"
    )

    return {
        "sample_times": sample_times,
        "raw_diameter": raw_diameter,
        "cleaned_diameter": cleaned,
        "artifact_mask": artifact_mask.astype(bool),
        "dlc_dataset": dlc_name,
        "times_dataset": time_name,
    }


def save_trace_cache(trace: dict[str, Any], path: Path, eid: str) -> Path:
    """Persist the numerical trace and JSON metadata in one compressed NPZ."""

    ensure_directory(path.parent)
    metadata = {
        "eid": str(eid),
        "camera": config.PUPIL_CAMERA,
        "likelihood_threshold": config.PUPIL_DLC_LIKELIHOOD_THRESHOLD,
        "blink_velocity_mad_threshold": config.BLINK_VELOCITY_MAD_THRESHOLD,
        "blink_padding_seconds": config.BLINK_PADDING_SECONDS,
        "frame_robust_z_max": config.FRAME_ROBUST_Z_MAX,
        "dlc_dataset": trace.get("dlc_dataset"),
        "times_dataset": trace.get("times_dataset"),
    }
    np.savez_compressed(
        path,
        sample_times=np.asarray(trace["sample_times"], dtype=float),
        raw_diameter=np.asarray(trace.get("raw_diameter", []), dtype=float),
        cleaned_diameter=np.asarray(trace["cleaned_diameter"], dtype=float),
        artifact_mask=np.asarray(trace["artifact_mask"], dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return path


def load_trace_cache(path: Path) -> dict[str, Any]:
    """Load a trace cache created by :func:`save_trace_cache`."""

    with np.load(path, allow_pickle=False) as cached:
        metadata_value = cached["metadata_json"]
        metadata = json.loads(str(metadata_value.item()))
        return {
            "sample_times": cached["sample_times"].astype(float),
            "raw_diameter": cached["raw_diameter"].astype(float),
            "cleaned_diameter": cached["cleaned_diameter"].astype(float),
            "artifact_mask": cached["artifact_mask"].astype(bool),
            **metadata,
        }


def load_or_fetch_trace(
    *,
    eid: str,
    cache_dir: Path,
    one: Any | None,
    force: bool,
    offline: bool,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], str]:
    """Return a trace and whether it came from cache or ONE."""

    path = _cache_path(eid, cache_dir)
    if path.exists() and not force:
        if progress is not None:
            size_mb = path.stat().st_size / (1024 ** 2)
            progress(f"loading cached trace {path.name} | size={size_mb:.1f} MB")
        trace = load_trace_cache(path)
        if progress is not None:
            cleaned = np.asarray(trace["cleaned_diameter"], dtype=float)
            missing = float(np.mean(~np.isfinite(cleaned))) if cleaned.size else np.nan
            missing_text = f"{missing:.1%}" if np.isfinite(missing) else "n/a"
            progress(
                f"cached trace loaded | samples={cleaned.size} | "
                f"cleaned missing={missing_text}"
            )
        return trace, "cache"
    if offline:
        raise FileNotFoundError(f"No cached pupil trace for {eid}: {path}")
    if one is None:
        raise RuntimeError("ONE client is required when a trace cache is absent.")
    if progress is not None:
        progress("cache miss; fetching trace from ONE")
    trace = fetch_and_clean_pupil_trace(one, eid, progress=progress)
    if progress is not None:
        progress(f"writing trace cache {path.name}")
    save_trace_cache(trace, path, eid)
    if progress is not None:
        size_mb = path.stat().st_size / (1024 ** 2)
        progress(f"trace cache written | size={size_mb:.1f} MB")
    return trace, "one"


def _window_feature(
    sample_times: np.ndarray,
    values: np.ndarray,
    event_time: float,
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
) -> dict[str, float | bool]:
    """Extract a complete-window baseline and baseline-subtracted response."""

    if not np.isfinite(event_time) or len(sample_times) < 2:
        return {"baseline": np.nan, "delta": np.nan, "valid": False}
    start_needed = event_time + min(baseline_window[0], response_window[0])
    stop_needed = event_time + max(baseline_window[1], response_window[1])
    if start_needed < sample_times[0] or stop_needed > sample_times[-1]:
        return {"baseline": np.nan, "delta": np.nan, "valid": False}

    baseline_mask = (
        (sample_times >= event_time + baseline_window[0])
        & (sample_times < event_time + baseline_window[1])
    )
    response_mask = (
        (sample_times >= event_time + response_window[0])
        & (sample_times <= event_time + response_window[1])
    )
    baseline_segment = values[baseline_mask]
    response_segment = values[response_mask]
    if baseline_segment.size == 0 or response_segment.size == 0:
        return {"baseline": np.nan, "delta": np.nan, "valid": False}

    baseline_fraction = np.isfinite(baseline_segment).mean()
    response_fraction = np.isfinite(response_segment).mean()
    valid = (
        baseline_fraction >= config.MIN_EVENT_WINDOW_VALID_FRACTION
        and response_fraction >= config.MIN_EVENT_WINDOW_VALID_FRACTION
    )
    if not valid:
        return {"baseline": np.nan, "delta": np.nan, "valid": False}
    baseline = float(np.nanmean(baseline_segment))
    response = float(np.nanmean(response_segment))
    if not (np.isfinite(baseline) and np.isfinite(response)):
        return {"baseline": np.nan, "delta": np.nan, "valid": False}
    return {"baseline": baseline, "delta": response - baseline, "valid": True}


def extract_session_features(session: pd.DataFrame, trace: dict[str, Any]) -> pd.DataFrame:
    """Extract all configured event-locked features for one session."""

    require_columns(
        session,
        [*config.TRIAL_KEY_COLUMNS, config.STIMULUS_TIME_COLUMN],
        table_name="session trial table",
    )
    sample_times = np.asarray(trace["sample_times"], dtype=float)
    values = np.asarray(trace["cleaned_diameter"], dtype=float)
    rows: list[dict[str, Any]] = []

    for _, trial in session.iterrows():
        stimulus = _window_feature(
            sample_times,
            values,
            float(trial.get(config.STIMULUS_TIME_COLUMN, np.nan)),
            config.STIMULUS_BASELINE_WINDOW,
            config.STIMULUS_PHASIC_WINDOW,
        )
        feedback = _window_feature(
            sample_times,
            values,
            float(trial.get(config.FEEDBACK_TIME_COLUMN, np.nan)),
            config.FEEDBACK_BASELINE_WINDOW,
            config.FEEDBACK_RESPONSE_WINDOW,
        )
        row = {column: trial[column] for column in config.TRIAL_KEY_COLUMNS}
        row.update(
            {
                config.PUPIL_TONIC_COLUMN: stimulus["baseline"],
                config.PUPIL_PHASIC_COLUMN: stimulus["delta"],
                config.PUPIL_FEEDBACK_PHASIC_COLUMN: feedback["delta"],
                "pupil_stimulus_window_valid": bool(stimulus["valid"]),
                "pupil_feedback_window_valid": bool(feedback["valid"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_metric_specific_qc(features: pd.DataFrame) -> pd.DataFrame:
    """Add session-context robust z-scores and independent validity flags."""

    result = features.copy()
    group_columns = [config.SUBJECT_COLUMN, config.SESSION_COLUMN]
    specifications = (
        (
            config.PUPIL_TONIC_COLUMN,
            config.PUPIL_TONIC_ROBUST_Z_COLUMN,
            config.TONIC_ROBUST_Z_MAX,
            "pupil_tonic_ok",
        ),
        (
            config.PUPIL_PHASIC_COLUMN,
            config.PUPIL_PHASIC_ROBUST_Z_COLUMN,
            config.PHASIC_ROBUST_Z_MAX,
            "pupil_phasic_ok",
        ),
        (
            config.PUPIL_FEEDBACK_PHASIC_COLUMN,
            config.PUPIL_FEEDBACK_PHASIC_ROBUST_Z_COLUMN,
            config.PHASIC_ROBUST_Z_MAX,
            "pupil_feedback_phasic_ok",
        ),
    )
    for raw_column, z_column, threshold, flag_column in specifications:
        result[z_column] = result.groupby(
            group_columns, sort=False, dropna=False
        )[raw_column].transform(lambda values: robust_zscore(values, constant="zeros"))
        result[flag_column] = (
            result[raw_column].notna() & result[z_column].abs().le(threshold)
        )
    return result


def build_qc_row(
    session: pd.DataFrame,
    trace: dict[str, Any] | None,
    *,
    source: str,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    """Create one session-level pupil QC record."""

    subject = str(session[config.SUBJECT_COLUMN].iloc[0])
    eid = str(session[config.SESSION_COLUMN].iloc[0])
    row: dict[str, Any] = {
        config.SUBJECT_COLUMN: subject,
        config.SESSION_COLUMN: eid,
        "trace_source": source,
        "status": status,
        "error": error,
        "n_trials": len(session),
        "excluded_subject": subject in config.PUPIL_SUBJECT_EXCLUSIONS,
        "exclusion_reason": config.PUPIL_SUBJECT_EXCLUSIONS.get(subject, ""),
    }
    if trace is None:
        row.update(
            n_samples=0,
            raw_missing_fraction=np.nan,
            cleaned_missing_fraction=np.nan,
            artifact_fraction=np.nan,
            trace_usable=False,
        )
        return row

    raw = np.asarray(trace.get("raw_diameter", []), dtype=float)
    cleaned = np.asarray(trace["cleaned_diameter"], dtype=float)
    artifact = np.asarray(trace["artifact_mask"], dtype=bool)
    missing = float(np.mean(~np.isfinite(cleaned))) if cleaned.size else np.nan
    row.update(
        n_samples=int(cleaned.size),
        raw_missing_fraction=(
            float(np.mean(~np.isfinite(raw))) if raw.size else np.nan
        ),
        cleaned_missing_fraction=missing,
        artifact_fraction=(float(artifact.mean()) if artifact.size else np.nan),
        trace_usable=bool(
            cleaned.size > 1
            and np.isfinite(missing)
            and missing <= config.MAX_PUPIL_NAN_FRACTION
        ),
    )
    return row


def diagnostic_plot(
    session: pd.DataFrame,
    trace: dict[str, Any],
    output_path: Path,
) -> Path:
    """Save a compact trace/window diagnostic for one representative session."""

    times = np.asarray(trace["sample_times"], dtype=float)
    diameter = np.asarray(trace["cleaned_diameter"], dtype=float)
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(times, diameter, linewidth=0.6)
    for event_time in pd.to_numeric(
        session[config.STIMULUS_TIME_COLUMN], errors="coerce"
    ).dropna().head(15):
        axis.axvspan(
            event_time + config.STIMULUS_BASELINE_WINDOW[0],
            event_time + config.STIMULUS_BASELINE_WINDOW[1],
            alpha=0.12,
        )
        axis.axvspan(
            event_time + config.STIMULUS_PHASIC_WINDOW[0],
            event_time + config.STIMULUS_PHASIC_WINDOW[1],
            alpha=0.12,
        )
    axis.set(
        xlabel="Session time (s)",
        ylabel="Cleaned pupil diameter",
        title=f"Pupil trace diagnostic: {session[config.SESSION_COLUMN].iloc[0]}",
    )
    figure.tight_layout()
    ensure_directory(output_path.parent)
    figure.savefig(output_path, dpi=config.FIGURE_DPI, facecolor="white")
    plt.close(figure)
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=config.TRIAL_TABLE_PATH)
    parser.add_argument("--output", type=Path, default=config.PUPIL_FEATURES_PATH)
    parser.add_argument("--cache-dir", type=Path, default=config.PUPIL_TRACE_CACHE_DIR)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true", help="Refresh existing caches.")
    parser.add_argument("--no-diagnostic-plot", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help=(
            "Show detailed within-session trace steps every N sessions. "
            "Session start/completion messages are always shown unless --no-progress is used."
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=60.0,
        help=(
            "During long blocking operations, emit a still-working heartbeat every "
            "N seconds. Use 0 to disable heartbeats. Default: 60."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress operator-facing progress messages while retaining warnings/errors.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1.")
    if args.heartbeat_seconds < 0:
        raise ValueError("--heartbeat-seconds cannot be negative.")
    config.ensure_project_directories()
    cache_dir = ensure_directory(args.cache_dir)
    logger = configure_stage_logger("02_preprocess_pupil", log_path=LOG_PATH)
    progress_enabled = not args.no_progress
    stage_started = time.monotonic()

    _progress(
        logger,
        progress_enabled,
        (
            "Stage 2 starting | pid=%d | log=%s | progress-every=%d | "
            "heartbeat=%.0fs"
        ),
        os.getpid(),
        LOG_PATH,
        args.progress_every,
        args.heartbeat_seconds,
    )
    _progress(logger, progress_enabled, "Loading trial table from %s", args.trials)
    trials = load_table(args.trials)
    require_unique_key(trials, config.TRIAL_KEY_COLUMNS, table_name="trial table")
    require_columns(
        trials,
        [config.STIMULUS_TIME_COLUMN, config.FEEDBACK_TIME_COLUMN],
        table_name="trial table",
    )
    total_sessions = int(trials[config.SESSION_COLUMN].nunique(dropna=False))
    total_subjects = int(trials[config.SUBJECT_COLUMN].nunique(dropna=False))
    _progress(
        logger,
        progress_enabled,
        (
            "Stage 2 input ready | trials=%d | sessions=%d | subjects=%d | "
            "cache=%s | offline=%s | force=%s"
        ),
        len(trials),
        total_sessions,
        total_subjects,
        cache_dir,
        args.offline,
        args.force,
    )
    one: Any | None = None
    if not args.offline:
        try:
            _progress(logger, progress_enabled, "Initializing ONE client")
            one = _create_one()
            _progress(logger, progress_enabled, "ONE client ready")
        except RuntimeError as error:
            logger.warning("ONE unavailable; only existing caches can be used: %s", error)
    else:
        _progress(logger, progress_enabled, "Offline mode enabled; using caches only")

    feature_pieces: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []
    first_diagnostic: tuple[pd.DataFrame, dict[str, Any]] | None = None
    counters = {
        "success": 0,
        "usable": 0,
        "rejected": 0,
        "failed": 0,
        "cache": 0,
        "one": 0,
        "trials": 0,
    }

    for session_index, (eid, session) in enumerate(
        trials.groupby(config.SESSION_COLUMN, sort=True),
        start=1,
    ):
        session_started = time.monotonic()
        session = session.sort_values(config.TRIAL_INDEX_COLUMN, kind="stable")
        subject = str(session[config.SUBJECT_COLUMN].iloc[0])
        cache_path = _cache_path(str(eid), cache_dir)
        expected_source = (
            "cache" if cache_path.exists() and not args.force else "ONE"
        )
        _progress(
            logger,
            progress_enabled,
            (
                "Session %d/%d | %s | %s | trials=%d | expected source=%s | starting"
            ),
            session_index,
            total_sessions,
            subject,
            eid,
            len(session),
            expected_source,
        )
        detailed_progress = (
            progress_enabled
            and (
                session_index == 1
                or session_index == total_sessions
                or session_index % args.progress_every == 0
            )
        )

        heartbeat_prefix = f"Session {session_index}/{total_sessions} | {subject} | {eid}"
        heartbeat = _SessionHeartbeat(
            logger,
            enabled=progress_enabled,
            interval_seconds=args.heartbeat_seconds,
            prefix=heartbeat_prefix,
        )

        def session_progress(message: str) -> None:
            heartbeat.update(message)
            _progress(
                logger,
                detailed_progress,
                "Session %d/%d | %s | %s | %s",
                session_index,
                total_sessions,
                subject,
                eid,
                message,
            )

        with heartbeat:
            try:
                trace, source = load_or_fetch_trace(
                    eid=str(eid),
                    cache_dir=cache_dir,
                    one=one,
                    force=args.force,
                    offline=args.offline or one is None,
                    progress=session_progress,
                )
                qc = build_qc_row(session, trace, source=source, status="success")
                if qc["trace_usable"]:
                    session_progress("extracting trial-level pupil features")
                    features = extract_session_features(session, trace)
                    stimulus_valid = float(features["pupil_stimulus_window_valid"].mean())
                    feedback_valid = float(features["pupil_feedback_window_valid"].mean())
                    session_progress(
                        "feature extraction complete | "
                        f"stimulus windows valid={stimulus_valid:.1%} | "
                        f"feedback windows valid={feedback_valid:.1%}"
                    )
                    if first_diagnostic is None:
                        first_diagnostic = (session, trace)
                else:
                    features = session[list(config.TRIAL_KEY_COLUMNS)].copy()
                    for column in (
                        config.PUPIL_TONIC_COLUMN,
                        config.PUPIL_PHASIC_COLUMN,
                        config.PUPIL_FEEDBACK_PHASIC_COLUMN,
                    ):
                        features[column] = np.nan
                    features["pupil_stimulus_window_valid"] = False
                    features["pupil_feedback_window_valid"] = False
                    qc["status"] = "rejected_missingness"
                    session_progress(
                        "trace rejected by cleaned-missingness QC | "
                        f"threshold={config.MAX_PUPIL_NAN_FRACTION:.1%}"
                    )
            except Exception as error:
                heartbeat.update(f"failed: {type(error).__name__}")
                logger.warning(
                    (
                        "Session %d/%d failed | subject=%s | eid=%s | "
                        "session elapsed=%s | error=%s: %s"
                    ),
                    session_index,
                    total_sessions,
                    subject,
                    eid,
                    _format_duration(time.monotonic() - session_started),
                    type(error).__name__,
                    error,
                )
                features = session[list(config.TRIAL_KEY_COLUMNS)].copy()
                for column in (
                    config.PUPIL_TONIC_COLUMN,
                    config.PUPIL_PHASIC_COLUMN,
                    config.PUPIL_FEEDBACK_PHASIC_COLUMN,
                ):
                    features[column] = np.nan
                features["pupil_stimulus_window_valid"] = False
                features["pupil_feedback_window_valid"] = False
                qc = build_qc_row(
                    session,
                    None,
                    source="none",
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
        feature_pieces.append(features)
        qc_rows.append(qc)

        counters["trials"] += len(session)
        status = str(qc["status"])
        if status == "success":
            counters["success"] += 1
        elif status == "rejected_missingness":
            counters["rejected"] += 1
        else:
            counters["failed"] += 1
        if bool(qc.get("trace_usable", False)):
            counters["usable"] += 1
        source = str(qc.get("trace_source", "none"))
        if source in counters:
            counters[source] += 1

        elapsed = time.monotonic() - stage_started
        average = elapsed / session_index
        eta = average * (total_sessions - session_index)
        missing = qc.get("cleaned_missing_fraction", np.nan)
        missing_text = f"{float(missing):.1%}" if np.isfinite(missing) else "n/a"
        _progress(
            logger,
            progress_enabled,
            (
                "Session %d/%d complete | status=%s | source=%s | usable=%s | "
                "missing=%s | session time=%s | cumulative trials=%d/%d | "
                "usable=%d | rejected=%d | failed=%d | elapsed=%s | ETA=%s"
            ),
            session_index,
            total_sessions,
            status,
            source,
            bool(qc.get("trace_usable", False)),
            missing_text,
            _format_duration(time.monotonic() - session_started),
            counters["trials"],
            len(trials),
            counters["usable"],
            counters["rejected"],
            counters["failed"],
            _format_duration(elapsed),
            _format_duration(eta),
        )

    _progress(
        logger,
        progress_enabled,
        "All sessions processed; concatenating %d feature pieces",
        len(feature_pieces),
    )
    features = pd.concat(feature_pieces, ignore_index=True)
    _progress(logger, progress_enabled, "Computing metric-specific robust-z QC")
    features = add_metric_specific_qc(features)
    require_unique_key(features, config.TRIAL_KEY_COLUMNS, table_name="pupil features")
    if len(features) != len(trials):
        raise AssertionError("Pupil feature table does not preserve one row per trial key.")

    qc_table = pd.DataFrame(qc_rows)
    exclusion_report = pd.DataFrame(
        [
            {config.SUBJECT_COLUMN: subject, "reason": reason}
            for subject, reason in config.PUPIL_SUBJECT_EXCLUSIONS.items()
        ]
    )
    _progress(logger, progress_enabled, "Saving pupil features to %s", args.output)
    save_table(features, args.output)
    _progress(logger, progress_enabled, "Saving pupil QC to %s", config.PUPIL_QC_PATH)
    save_table(qc_table, config.PUPIL_QC_PATH)
    _progress(
        logger,
        progress_enabled,
        "Saving pupil exclusion report to %s",
        config.PUPIL_EXCLUSION_REPORT_PATH,
    )
    save_table(exclusion_report, config.PUPIL_EXCLUSION_REPORT_PATH)

    metadata = {
        "stimulus_tonic": {
            "event": config.STIMULUS_TIME_COLUMN,
            "window": config.STIMULUS_BASELINE_WINDOW,
            "column": config.PUPIL_TONIC_COLUMN,
        },
        "stimulus_phasic": {
            "event": config.STIMULUS_TIME_COLUMN,
            "baseline_window": config.STIMULUS_BASELINE_WINDOW,
            "response_window": config.STIMULUS_PHASIC_WINDOW,
            "column": config.PUPIL_PHASIC_COLUMN,
        },
        "feedback_phasic": {
            "event": config.FEEDBACK_TIME_COLUMN,
            "baseline_window": config.FEEDBACK_BASELINE_WINDOW,
            "response_window": config.FEEDBACK_RESPONSE_WINDOW,
            "column": config.PUPIL_FEEDBACK_PHASIC_COLUMN,
        },
        "minimum_valid_fraction": config.MIN_EVENT_WINDOW_VALID_FRACTION,
        "trace_cache_directory": str(cache_dir.resolve()),
    }
    _progress(
        logger,
        progress_enabled,
        "Writing pupil feature metadata to %s",
        config.PUPIL_FEATURE_METADATA_PATH,
    )
    config.PUPIL_FEATURE_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    if first_diagnostic is not None and not args.no_diagnostic_plot:
        _progress(logger, progress_enabled, "Saving representative diagnostic plot")
        diagnostic_plot(
            *first_diagnostic,
            config.FIGURE_DIR / "pupil_trace_window_diagnostic.png",
        )

    logger.info(
        (
            "Stage 2 complete: %d trials, %d sessions, %d usable traces | "
            "success=%d | rejected=%d | failed=%d | cache=%d | ONE=%d | runtime=%s"
        ),
        len(features),
        len(qc_table),
        int(qc_table.get("trace_usable", pd.Series(dtype=bool)).sum()),
        counters["success"],
        counters["rejected"],
        counters["failed"],
        counters["cache"],
        counters["one"],
        _format_duration(time.monotonic() - stage_started),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
