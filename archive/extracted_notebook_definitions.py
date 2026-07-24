

# CELL 3 :: GLMHMM
class GLMHMM:
    """
    Bernoulli-GLM-HMM fit by EM (Ashwood et al. 2022), MAP with Gaussian prior
    on W (variance W_PRIOR_VAR) and sticky-Dirichlet prior on A
    (A_STICKY_ALPHA).
    """
    def __init__(self, n_states=3, n_features=4, seed=0):
        self.K, self.D = n_states, n_features
        self.rng = np.random.default_rng(seed)
        self.W = self.A = self.pi = None
        self.ll_history = []
        self._lr_cache = [None] * self.K

    def _log_emission(self, X, y):
        p1 = np.clip(sigmoid(X @ self.W.T), 1e-9, 1 - 1e-9)  # (T, K)
        return y[:, None] * np.log(p1) + (1 - y[:, None]) * np.log(1 - p1)

    def _forward_backward(self, log_em):
        T, K = log_em.shape
        logA = np.log(self.A + 1e-16)
        logpi = np.log(self.pi + 1e-16)

        la = np.zeros((T, K))
        la[0] = logpi + log_em[0]
        for t in range(1, T):
            la[t] = log_em[t] + logsumexp(la[t - 1][:, None] + logA, axis=0)

        lb = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(
                logA + log_em[t + 1][None, :] + lb[t + 1][None, :],
                axis=1
            )

        ll = logsumexp(la[-1])
        lg = la + lb
        lg -= logsumexp(lg, axis=1, keepdims=True)
        gamma = np.exp(lg)

        xi = np.zeros((K, K))
        for t in range(T - 1):
            m = (
                la[t][:, None]
                + logA
                + log_em[t + 1][None, :]
                + lb[t + 1][None, :]
            )
            xi += np.exp(m - logsumexp(m))

        return gamma, xi, ll

    def _init(self):
        K, D = self.K, self.D
        self.A = np.full((K, K), 0.02 / max(K - 1, 1))
        np.fill_diagonal(self.A, 0.98)
        self.pi = np.full(K, 1 / K)
        self.W = 0.2 * self.rng.standard_normal((K, D))
        self.W[0, 0] = 3.0  # seed engaged state
        self.ll_history = []
        self._lr_cache = [None] * self.K

    def _m_step_weights(self, X, y, gamma):
        y_int = np.asarray(y, dtype=int)
        for k in range(self.K):
            w = gamma[:, k]
            if w.sum() < 1e-8:
                continue
            if (w * y).sum() < 1e-6 or (w * (1 - y)).sum() < 1e-6:
                continue

            lr = self._lr_cache[k]
            if lr is None:
                lr = LogisticRegression(
                    C=W_PRIOR_VAR,
                    fit_intercept=False,
                    solver='lbfgs',
                    penalty='l2',
                    warm_start=True,
                    max_iter=LR_MAX_ITER,
                    tol=LR_TOL,
                )
                self._lr_cache[k] = lr

            lr.fit(X, y_int, sample_weight=w)
            self.W[k] = lr.coef_[0]

    def fit(
        self, X, y, n_iter=150, tol=1e-4, rel_tol=1e-4,
        patience=3, min_iter=10, verbose=False
    ):
        self._init()
        self.ll_history = []
        prev = -np.inf
        stall = 0
        y = np.asarray(y, dtype=float)

        for it in range(n_iter):
            gamma, xi, ll = self._forward_backward(self._log_emission(X, y))
            self.ll_history.append(ll)

            self.pi = gamma[0] / gamma[0].sum()
            prior_counts = np.eye(self.K) * (A_STICKY_ALPHA - 1)
            post = xi + prior_counts
            self.A = post / post.sum(axis=1, keepdims=True)

            # Patched: use cached / warm-start weighted M-step instead of
            # re-instantiating LogisticRegression for every state every EM iter.
            self._m_step_weights(X, y, gamma)

            if verbose and (it % 25 == 0 or it == n_iter - 1):
                print(f' EM iter {it:3d} loglik={ll:.2f}')

            if it >= min_iter and np.isfinite(prev):
                gain = ll - prev
                rel_gain = gain / max(abs(prev), 1e-12)
                if gain < tol or rel_gain < rel_tol:
                    stall += 1
                else:
                    stall = 0
                if stall >= patience:
                    if verbose:
                        print(
                            f' early stop @ iter {it:3d}: '
                            f'gain={gain:.3e}, rel_gain={rel_gain:.3e}, stall={stall}'
                        )
                    break
            prev = ll

        return self

    def posterior(self, X, y):
        return self._forward_backward(self._log_emission(X, y))[0]

    def loglik(self, X, y):
        return self._forward_backward(self._log_emission(X, y))[2]


# CELL 3 :: fit_best
def fit_best(
    n_states, n_features, X, y, n_init=N_INIT, base_seed=0,
    verbose=False, screen_iters=SCREEN_ITERS, keep_top=KEEP_TOP_INITS, **fit_kw
):
    """
    Minimal speed patch:
      1) run a short EM screen on all random starts,
      2) fully fit only the top-ranked starts.

    Keeps behavior close to the original multi-start strategy while avoiding
    n_init full EM runs when most starts are clearly suboptimal early.
    """
    n_init = int(n_init)
    keep_top = max(1, min(int(keep_top), n_init))

    if n_init == 1:
        return GLMHMM(n_states, n_features, seed=base_seed).fit(
            X, y, verbose=verbose, **fit_kw
        )

    screen_scores = []
    for i in range(n_init):
        seed = base_seed + i
        m = GLMHMM(n_states, n_features, seed=seed).fit(
            X, y, n_iter=screen_iters, tol=1e-3, verbose=False
        )
        ll = m.ll_history[-1] if m.ll_history else -np.inf
        screen_scores.append((ll, seed))
        if verbose:
            print(f' screen init {i:2d}/{n_init}: ll={ll:.2f}')

    screen_scores.sort(key=lambda z: z[0], reverse=True)
    kept_seeds = [seed for _, seed in screen_scores[:keep_top]]

    best_model, best_ll = None, -np.inf
    for j, seed in enumerate(kept_seeds, start=1):
        m = GLMHMM(n_states, n_features, seed=seed).fit(
            X, y, verbose=False, **fit_kw
        )
        final_ll = m.ll_history[-1] if m.ll_history else -np.inf
        if verbose:
            print(
                f' full init {j:2d}/{keep_top}: '
                f'seed={seed}, final train loglik={final_ll:.2f}'
            )
        if final_ll > best_ll:
            best_model, best_ll = m, final_ll

    if verbose:
        print(f' -> kept init with train loglik={best_ll:.2f}')
    return best_model


# CELL 3 :: relabel_states_by_engagement
def relabel_states_by_engagement(model):
    """
    Order/label the 3 states the Ashwood (2022) way.
    Design cols = [stim_contrast, bias, prev_choice, prev_stim];
    y=1 = RIGHTWARD.
    """
    stim = np.abs(model.W[:, 0])
    bias = model.W[:, 1]
    engaged = int(np.argmax(stim))
    rest = [k for k in range(model.K) if k != engaged]
    rest_sorted = sorted(rest, key=lambda k: bias[k])  # ascending: left first
    biased_left, biased_right = rest_sorted[0], rest_sorted[1]
    order = np.array([engaged, biased_left, biased_right])
    labels = ['engaged', 'biased-left', 'biased-right']
    return order, labels


# CELL 3 :: order_by_engagement
def order_by_engagement(model):
    return relabel_states_by_engagement(model)[0]


# CELL 5 :: find_video_sessions
def find_video_sessions(subject):
    '''eids for a subject that have a left-camera VIDEO (raw movie).'''
    try:
        return list(one.search(subject=subject,
                               datasets=['_iblrig_leftCamera.raw.mp4'],
                               query_type='remote'))
    except Exception:
        return []


# CELL 5 :: is_choiceworld
def is_choiceworld(eid):
    '''True only for the 2AFC ChoiceWorld task (ephys/biased/training) - excludes
    passive replay, spontaneous, and any non-2AFC protocol so no non-behaviour
    session can leak into the roster.'''
    try:
        p = one.get_details(str(eid)).get('task_protocol', '') or ''
    except Exception:
        return False
    p = p.lower()
    if any(b in p for b in ('passive', 'replay', 'spontaneous', 'habituation')):
        return False
    return 'choiceworld' in p


# CELL 5 :: find_pupil_sessions
def find_pupil_sessions(subject):
    '''eids for a subject that have precomputed left-camera DLC *and* are the 2AFC
    ChoiceWorld task. Protocol is verified per session, not assumed.'''
    try:
        eids = one.search(subject=subject,
                          datasets=['_ibl_leftCamera.dlc.pqt'],
                          query_type='remote')
    except Exception:
        return []
    return [str(e) for e in eids if is_choiceworld(e)]


# CELL 5 :: get_sex
def get_sex(subject):
    try:
        return one.alyx.rest('subjects', 'read', id=subject).get('sex')
    except Exception:
        return None


# CELL 5 :: load_session_trials
def load_session_trials(eid):
    '''One session -> trial DataFrame with signed_contrast + accuracy.'''
    sl = SessionLoader(eid=eid, one=one); sl.load_trials()
    tr = sl.trials.copy().reset_index(drop=True)
    cl, cr = tr['contrastLeft'].to_numpy(), tr['contrastRight'].to_numpy()
    tr['signed_contrast'] = np.nan_to_num(cr, nan=0.) - np.nan_to_num(cl, nan=0.)
    tr['accuracy'] = (tr['feedbackType'] == 1).astype(int)
    return tr


# CELL 10 :: label_trial_epochs
def label_trial_epochs(trials_df, n_transition_trials=10):
    df = trials_df.copy().reset_index(drop=True)
    pL = df['probabilityLeft'].to_numpy()
    epoch = np.array(['stable']*len(df), dtype=object)
    epoch[pL == 0.5] = 'unbiased'
    changed = np.zeros(len(df), bool); changed[1:] = pL[1:] != pL[:-1]
    for t in np.where(changed)[0]:
        if pL[t] == 0.5:
            continue
        for j in range(t, min(t+n_transition_trials, len(df))):
            if epoch[j] == 'stable':
                epoch[j] = 'transition'
    df['epoch'] = epoch
    return df


# CELL 12 :: build_design_matrix
def build_design_matrix(df):
    '''Ashwood inputs: [z(signed_contrast), bias, prev_choice(+-1), prev_stim(+-1)], y=rightward.'''
    df = df.reset_index(drop=True)
    sc = df['signed_contrast'].to_numpy(float)
    sd = np.nanstd(sc); scz = sc/sd if sd > 0 else sc
    y = rightward_choice(df)
    pc = np.zeros(len(df)); pc[1:] = np.where(y[:-1] == 1, 1., -1.)
    ps = np.zeros(len(df)); ps[1:] = np.sign(sc[:-1])
    X = np.column_stack([scz, np.ones(len(df)), pc, ps])
    return X, y


# CELL 12 :: fit_animal_glmhmm
def fit_animal_glmhmm(trials_list, n_states=N_STATES, seed=0, verbose=False,
                      test_frac=0.25):
    '''trials_list: list of per-session epoch-labelled DataFrames.
    Cross-validation is done at the SESSION level (whole sessions held out), not
    by random trials: trials within a session are temporally dependent, so a
    random-trial split would leak information across train/test. We fit EM on the
    training sessions only, then score held-out sessions with the frozen model and
    report per-trial held-out log-likelihood (the standard GLM-HMM fit metric).'''
    # per-session design matrices
    keep_all, X_all, y_all = [], [], []
    for df in trials_list:
        d = df[df['choice'] != 0].reset_index(drop=True)
        X, y = build_design_matrix(d)
        keep_all.append(d); X_all.append(X); y_all.append(y)
    n_sess = len(keep_all)

    # choose held-out sessions (>=1 test session when there are >=2 sessions)
    rng = np.random.default_rng(seed)
    if n_sess >= 2 and test_frac > 0:
        n_test = max(1, int(round(test_frac * n_sess)))
        test_idx = set(rng.choice(n_sess, size=n_test, replace=False).tolist())
    else:
        test_idx = set()  # too few sessions to hold any out
    train_idx = [i for i in range(n_sess) if i not in test_idx]

    # fit on TRAIN sessions only
    Xtr = np.vstack([X_all[i] for i in train_idx])
    ytr = np.concatenate([y_all[i] for i in train_idx])
    model = fit_best(n_states, Xtr.shape[1], Xtr, ytr, n_init=N_INIT, base_seed=seed, verbose=verbose)  # multi-init, was single-seed .fit()
    order, labels = relabel_states_by_engagement(model)

    # held-out score: per-trial log-likelihood on train vs test (frozen model)
    def _per_trial_ll(idxs):
        if not idxs:
            return np.nan, 0
        ll = sum(model.loglik(X_all[i], y_all[i]) for i in idxs)
        n  = sum(len(y_all[i]) for i in idxs)
        return (ll / n if n else np.nan), n
    tr_ll, n_tr = _per_trial_ll(train_idx)
    te_ll, n_te = _per_trial_ll(sorted(test_idx))
    model.cv_ = dict(train_ll=tr_ll, test_ll=te_ll, n_train=n_tr, n_test=n_te,
                     n_sess=n_sess, n_test_sess=len(test_idx))
    if verbose:
        print(f'  held-out: {len(test_idx)}/{n_sess} sessions | '
              f'train LL/trial={tr_ll:.4f} (n={n_tr}), '
              f'test LL/trial={te_ll:.4f} (n={n_te})')
    # posteriors per session (respect trial order), then reattach.
    # NOTE: posteriors/states are attached to ALL sessions (train + held-out) so
    # the pupil analysis uses every trial. The holdout only feeds the fit-quality
    # (held-out LL) metric above; it does not discard data from the pupil tests.
    out = []
    for d in keep_all:
        Xi, yi = build_design_matrix(d)
        g = model.posterior(Xi, yi)[:, order]
        d = d.copy()
        for k in range(n_states):
            d[f'p_state{k}'] = g[:, k]
        d['state'] = np.argmax(g, axis=1)
        # readable Ashwood labels: 0=engaged, 1=biased-left, 2=biased-right
        d['state_label'] = [labels[s] for s in d['state']]
        out.append(d)
    return model, pd.concat(out, ignore_index=True), order


# CELL 15 :: remove_blinks
def remove_blinks(diam, times, vel_sd=BLINK_VEL_SD, pad_s=BLINK_PAD_S):
    # Explicit blink/artifact removal AFTER IBL smoothing, following pupillometry
    # best practice (Kret & Sjak-Shie 2019; Mathot 2018):
    #   1) any frame already NaN (failed likelihood/IBL clip) is a candidate,
    #   2) frames whose |velocity| (d diam / dt) exceeds vel_sd robust-SDs are
    #      flagged -- blink onsets/offsets are high-velocity edges,
    #   3) flagged frames are dilated by +-pad_s (~125 ms) so the partial-occlusion
    #      frames flanking a blink are also removed.
    # Returns a copy of diam with artifact frames set to NaN (NOT interpolated --
    # downstream uses nan-aware means so gaps simply don't contribute).
    d = np.asarray(diam, float).copy()
    t = np.asarray(times, float)
    bad = ~np.isfinite(d)
    dt = np.gradient(t)
    dt[dt <= 0] = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0/FPS
    vel = np.abs(np.gradient(d) / dt)                 # units: px/s
    finite_vel = vel[np.isfinite(vel)]
    if finite_vel.size:
        med = np.nanmedian(finite_vel)
        mad = np.nanmedian(np.abs(finite_vel - med)) + 1e-9
        rob_sd = 1.4826 * mad                          # robust SD via MAD
        bad |= vel > (med + vel_sd * rob_sd)
    # dilate bad mask by +-pad frames
    pad = int(round(pad_s * FPS))
    if pad > 0 and bad.any():
        idx = np.where(bad)[0]
        for j in idx:
            bad[max(0, j-pad):min(len(bad), j+pad+1)] = True
    d[bad] = np.nan
    return d


# CELL 15 :: load_pupil_diameter
def load_pupil_diameter(eid, cam=CAM, lik=LIK_THRESH):
    # Return (blink-cleaned_diameter, cam_times) or (None, None). Handles revisions.
    ds = one.list_datasets(eid, filename=f'*{cam}Camera*', query_type='remote')
    dlc_ds = [d for d in ds if f'{cam}Camera.dlc' in d]
    t_ds   = [d for d in ds if f'{cam}Camera.times' in d]
    if not dlc_ds or not t_ds:
        return None, None
    dlc = one.load_dataset(eid, dlc_ds[0])
    cam_times = one.load_dataset(eid, t_ds[0])
    diam = get_smooth_pupil_diameter(
    get_pupil_diameter(
        likelihood_threshold(
            dlc,
            threshold=lik,
        )
    ),
    cam,
    std_thresh=5,
    nan_thresh=1,
)

    diam = remove_blinks(
        diam,
        cam_times,
    )

    # Remove nonphysical or extreme tracking plateaus that may not have
    # high-velocity blink edges.
    finite = np.isfinite(diam)

    if finite.any():
        session_median = np.nanmedian(diam)

        session_mad = np.nanmedian(
            np.abs(
                diam[finite]
                - session_median
            )
        )

        session_scale = 1.4826 * session_mad

        if (
            np.isfinite(session_scale)
            and session_scale
            > np.finfo(float).eps
        ):
            frame_rz = (
                diam
                - session_median
            ) / session_scale

            diam[
                np.abs(frame_rz) > 8.0
            ] = np.nan

    # Diameter cannot legitimately be zero or negative.
    diam[
        np.isfinite(diam)
        & (diam <= 0)
    ] = np.nan

    return diam, cam_times


# CELL 15 :: per_trial_pupil
def per_trial_pupil(
    diam,
    cam_times,
    df,
    base_win=BASELINE_WIN,
    evk_win=EVOKED_WIN,
    min_valid_fraction=0.80,
):
    """
    Extract trial-level tonic and phasic pupil measurements.

    tonic:
        Mean diameter in the prestimulus baseline window.

    phasic:
        Mean diameter in the evoked window minus the trial's baseline.

    A trial is retained only when:
        1. The complete requested window lies inside the camera recording.
        2. At least min_valid_fraction of the expected frames are finite.
        3. The resulting means are finite.
    """
    diam = np.asarray(diam, dtype=float)
    cam_times = np.asarray(cam_times, dtype=float)

    if diam.ndim != 1 or cam_times.ndim != 1:
        raise ValueError("diam and cam_times must be one-dimensional.")

    if len(diam) != len(cam_times):
        raise ValueError(
            f"diam and cam_times differ in length: "
            f"{len(diam)} versus {len(cam_times)}"
        )

    if len(cam_times) < 2:
        return (
            np.full(len(df), np.nan),
            np.full(len(df), np.nan),
        )

    valid_dt = np.diff(cam_times)
    valid_dt = valid_dt[
        np.isfinite(valid_dt) & (valid_dt > 0)
    ]

    if valid_dt.size == 0:
        return (
            np.full(len(df), np.nan),
            np.full(len(df), np.nan),
        )

    median_dt = np.median(valid_dt)

    baseline_duration = base_win[1] - base_win[0]
    evoked_duration = evk_win[1] - evk_win[0]

    expected_baseline_frames = max(
        1,
        int(np.floor(baseline_duration / median_dt)),
    )

    expected_evoked_frames = max(
        1,
        int(np.floor(evoked_duration / median_dt)),
    )

    min_baseline_frames = max(
        1,
        int(
            np.ceil(
                min_valid_fraction
                * expected_baseline_frames
            )
        ),
    )

    min_evoked_frames = max(
        1,
        int(
            np.ceil(
                min_valid_fraction
                * expected_evoked_frames
            )
        ),
    )

    stim_times = pd.to_numeric(
        df["stimOn_times"],
        errors="coerce",
    ).to_numpy(dtype=float)

    tonic = np.full(len(df), np.nan, dtype=float)
    phasic = np.full(len(df), np.nan, dtype=float)

    recording_start = cam_times[0]
    recording_end = cam_times[-1]

    for trial_index, stim_time in enumerate(stim_times):
        if not np.isfinite(stim_time):
            continue

        baseline_start = stim_time + base_win[0]
        baseline_end = stim_time + base_win[1]
        evoked_start = stim_time + evk_win[0]
        evoked_end = stim_time + evk_win[1]

        # Reject partial windows at recording boundaries.
        if (
            baseline_start < recording_start
            or baseline_end > recording_end
            or evoked_start < recording_start
            or evoked_end > recording_end
        ):
            continue

        baseline_i0, baseline_i1 = np.searchsorted(
            cam_times,
            [baseline_start, baseline_end],
        )

        evoked_i0, evoked_i1 = np.searchsorted(
            cam_times,
            [evoked_start, evoked_end],
        )

        baseline_segment = diam[
            baseline_i0:baseline_i1
        ]

        evoked_segment = diam[
            evoked_i0:evoked_i1
        ]

        baseline_finite = np.isfinite(
            baseline_segment
        )

        evoked_finite = np.isfinite(
            evoked_segment
        )

        baseline_ok = (
            baseline_segment.size > 0
            and baseline_finite.sum()
            >= min_baseline_frames
        )

        evoked_ok = (
            evoked_segment.size > 0
            and evoked_finite.sum()
            >= min_evoked_frames
        )

        if not baseline_ok:
            continue

        baseline_mean = np.nanmean(
            baseline_segment
        )

        if not np.isfinite(baseline_mean):
            continue

        tonic[trial_index] = baseline_mean

        if evoked_ok:
            evoked_mean = np.nanmean(
                evoked_segment
            )

            if np.isfinite(evoked_mean):
                phasic[trial_index] = (
                    evoked_mean
                    - baseline_mean
                )

    return tonic, phasic


# CELL 48 :: relabel_states_by_engagement
def relabel_states_by_engagement(model):
    stim = np.abs(model.W[:, 0])
    bias = model.W[:, 1]

    engaged = int(np.argmax(stim))
    rest = [k for k in range(model.K) if k != engaged]
    rest_sorted = sorted(rest, key=lambda k: bias[k])  # more negative = left
    biased_left, biased_right = rest_sorted[0], rest_sorted[1]

    order = np.array([engaged, biased_left, biased_right])
    labels = ['engaged', 'biased-left', 'biased-right']
    return order, labels


# CELL 51 :: sanitize_probability_matrix
def sanitize_probability_matrix(raw_probabilities):
    """
    Convert an N x K array into valid probability vectors.

    Operations:
        - require finite values;
        - clip numerical negative values to zero;
        - renormalize after clipping;
        - mark rows with zero total probability as invalid.
    """
    raw = np.asarray(
        raw_probabilities,
        dtype=float,
    )

    if raw.ndim != 2:
        raise ValueError(
            "Probability matrix must be two-dimensional."
        )

    finite_rows = np.isfinite(raw).all(axis=1)

    cleaned = np.full_like(
        raw,
        np.nan,
        dtype=float,
    )

    cleaned[finite_rows] = np.clip(
        raw[finite_rows],
        0.0,
        None,
    )

    row_sums = np.nansum(
        cleaned,
        axis=1,
    )

    valid_rows = (
        finite_rows
        & np.isfinite(row_sums)
        & (row_sums > 0)
    )

    normalized = np.full_like(
        cleaned,
        np.nan,
        dtype=float,
    )

    normalized[valid_rows] = (
        cleaned[valid_rows]
        / row_sums[
            valid_rows,
            None,
        ]
    )

    return normalized, valid_rows


# CELL 52 :: robust_zscore_series
def robust_zscore_series(series):
    """
    Robust within-group z-score using median and MAD.
    Falls back to ordinary SD when MAD is zero.
    """
    x = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    center = x.median(skipna=True)

    mad = (
        x - center
    ).abs().median(skipna=True)

    scale = 1.4826 * mad

    if (
        not np.isfinite(scale)
        or scale <= np.finfo(float).eps
    ):
        scale = x.std(
            skipna=True,
            ddof=0,
        )

    if (
        not np.isfinite(scale)
        or scale <= np.finfo(float).eps
    ):
        return pd.Series(
            np.nan,
            index=x.index,
            dtype=float,
        )

    return (
        x - center
    ) / scale


# CELL 53 :: select_isolated_bursts
def select_isolated_bursts(sequence):
    candidate_positions = np.flatnonzero(
        sequence["burst_candidate"]
        .to_numpy(dtype=bool)
    )

    if len(candidate_positions) == 0:
        return []

    amplitudes = (
        sequence["phasic_for_burst"]
        .to_numpy(dtype=float)
    )

    # Examine the largest candidates first.
    ordered_candidates = sorted(
        candidate_positions,
        key=lambda position: amplitudes[position],
        reverse=True,
    )

    accepted = []

    for position in ordered_candidates:
        too_close = any(
            abs(position - selected_position)
            <= REFRACTORY_TRIALS
            for selected_position in accepted
        )

        if not too_close:
            accepted.append(position)

    return sorted(accepted)
