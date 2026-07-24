

# ===== CELL 1 =====
PIPELINE_STATE = {
    "stage": "QC",
    "inputs": None,
    "outputs": None,
    "open_issue": None,
}

def set_pipeline_state(stage, inputs=None, outputs=None, open_issue=None):
    PIPELINE_STATE["stage"] = stage
    PIPELINE_STATE["inputs"] = inputs
    PIPELINE_STATE["outputs"] = outputs
    PIPELINE_STATE["open_issue"] = open_issue
    print("STATE:", PIPELINE_STATE["stage"])
    if inputs is not None:
        print("INPUTS:", inputs)
    if outputs is not None:
        print("OUTPUTS:", outputs)
    if open_issue is not None:
        print("OPEN:", open_issue)



# ===== CELL 2 =====
# %pip install ONE-api ibllib scikit-learn scipy matplotlib pandas numpy
import os, warnings
os.environ.setdefault('ONE_HTTP_DL_THREADS', '1')
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.special import logsumexp, expit as sigmoid
from scipy.stats import spearmanr, kruskal, mannwhitneyu, wilcoxon
from sklearn.linear_model import LogisticRegression

from one.api import ONE
from brainbox.io.one import SessionLoader
from brainbox.behavior.dlc import (
    likelihood_threshold, get_pupil_diameter, get_smooth_pupil_diameter,
)

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

LIK_THRESH = 0.9     # DLC likelihood threshold (IBL convention)
NAN_GATE   = 0.30    # reject a pupil trace if > this fraction is NaN
N_STATES   = 3       # Ashwood: engaged + 2 biased/disengaged
CAM        = 'left'  # left camera has the eye used for pupil
FPS        = 60.0    # IBL left-camera nominal frame rate (used for velocity units)
BLINK_PAD_S   = 0.125  # blink padding each side (~125 ms; Kret & Sjak-Shie 2019)
BLINK_VEL_SD  = 5.0    # velocity-based blink/outlier threshold (SD of |d diam/dt|)
BASELINE_WIN  = (-0.5, 0.0)  # pre-stimOn tonic baseline window (s)
EVOKED_WIN    = (0.0, 1.0)   # post-stimOn phasic/evoked window (s)
rng = np.random.default_rng(0)
print('connected')



# ===== CELL 3 =====
# --- MAP priors + minimal speed patch for GLM-HMM fitting ---
# Keeps the notebook interface stable:
# - same GLMHMM class name
# - same fit_best(...) signature (plus optional speed kwargs)
# - same relabel/order helpers and STATE_LABELS

W_PRIOR_VAR = 2.0      # sigma^2, Ashwood's CV-selected value for IBL mice
A_STICKY_ALPHA = 2.0   # sticky Dirichlet concentration on self-transitions
N_INIT = 20

# Conservative speed defaults; override per call if needed.
SCREEN_ITERS = 20      # short EM run for ranking random starts
KEEP_TOP_INITS = 4     # fully fit only the best few starts
LR_MAX_ITER = 150      # lighter than original 500
LR_TOL = 1e-4          # lighter than original 1e-6


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


def order_by_engagement(model):
    return relabel_states_by_engagement(model)[0]


STATE_LABELS = ['engaged', 'biased-left', 'biased-right']



# ===== CELL 4 =====
W_true = np.array([[4.,0.,0.,0.],[.3,2.,0.,0.],[.3,-2.,0.,0.]])
A_true = np.array([[.97,.015,.015],[.02,.96,.02],[.02,.02,.96]])
pi_true = np.array([.6,.2,.2]); Tn = 4000
c = rng.choice([-1,-.5,-.25,-.125,0,.125,.25,.5,1], size=Tn)
Xs = np.column_stack([c, np.ones(Tn), rng.choice([-1,1],Tn), rng.choice([-1,1],Tn)])
z = np.zeros(Tn, int); z[0] = rng.choice(3, p=pi_true)
for t in range(1, Tn):
    z[t] = rng.choice(3, p=A_true[z[t-1]])
ys = (rng.random(Tn) < sigmoid(np.sum(Xs*W_true[z], axis=1))).astype(float)

ms = fit_best(3, 4, Xs, ys, n_init=N_INIT, base_seed=1, verbose=True)  # multi-init, was single-seed .fit()
o = order_by_engagement(ms)
print('\nrecovered W (engaged-first):\n', np.round(ms.W[o], 2))
print('true W:\n', np.round(W_true, 2))
zc = o[np.argmax(ms.posterior(Xs, ys), axis=1)]
best = max(permutations(range(3)), key=lambda pm: np.mean([pm[a]==b for a,b in zip(zc,z)]))
acc = np.mean([best[a]==b for a,b in zip(zc, z)])
print(f'\nstate-decode accuracy: {acc:.3f}  ->', 'PASS' if acc>0.8 else 'CHECK')



# ===== CELL 5 =====
def find_video_sessions(subject):
    '''eids for a subject that have a left-camera VIDEO (raw movie).'''
    try:
        return list(one.search(subject=subject,
                               datasets=['_iblrig_leftCamera.raw.mp4'],
                               query_type='remote'))
    except Exception:
        return []

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

def get_sex(subject):
    try:
        return one.alyx.rest('subjects', 'read', id=subject).get('sex')
    except Exception:
        return None

def load_session_trials(eid):
    '''One session -> trial DataFrame with signed_contrast + accuracy.'''
    sl = SessionLoader(eid=eid, one=one); sl.load_trials()
    tr = sl.trials.copy().reset_index(drop=True)
    cl, cr = tr['contrastLeft'].to_numpy(), tr['contrastRight'].to_numpy()
    tr['signed_contrast'] = np.nan_to_num(cr, nan=0.) - np.nan_to_num(cl, nan=0.)
    tr['accuracy'] = (tr['feedbackType'] == 1).astype(int)
    return tr



# ===== CELL 6 =====
# ---- EXPLICIT SEX-BALANCED ALLOWLIST (ephysChoiceWorld + DLC pupil) ----
# 18 F + 18 M = 36 animals. Males matched to females by pupil-session count.
# All are 2AFC ChoiceWorld animals with left-camera DLC pupil (verified in the
# 95-animal sweep). Pinning the roster makes the sample fully reproducible and
# skips server-wide discovery.
SUBJECT_ALLOWLIST = [None
    # --- females (all 18 available in the cohort) ---
    #'CSHL024', 'CSHL025', 'CSHL051', 'CSHL052', 'CSHL053', 'CSHL055',
    #'CSHL068', 'CSHL069', 'CSP015',  'FD_24',   'FD_28',   'FMR028',
    #'KS023',   'NR_0024', 'NR_0028', 'NR_0029', 'NR_0031', 'NYU-12',
    # --- males (18, matched to females by pupil-session count) ---
    #'CSHL045', 'CSHL046', 'CSHL047', 'CSHL049', 'CSHL058', 'CSHL059',
    #'CSHL060', 'CSHL072', 'CSP023',  'DY_009',  'DY_010',  'FMR019',
    #'KS014',   'NR_0027', 'NR_0019', 'NYU-31',  'NYU-38',  'NYU-40',
]

#roster = list(SUBJECT_ALLOWLIST)
#TARGET_N = len(roster)
#n_f = sum(1 for s in roster if s in SUBJECT_ALLOWLIST[:18])
#print(f'sex-balanced allowlist: {n_f} F + {TARGET_N - n_f} M = {TARGET_N} animals')
#print('roster:', roster)



# ===== CELL 7 =====
# ---- FULL SWEEP: discover all DLC-capable subjects from the server ----
from tqdm.auto import tqdm

SUBJECT_ALLOWLIST = None
MIN_DLC_SESSIONS = 2          # inclusion rule: keep subjects with >= 2 DLC sessions
MIN_TOTAL_TRIALS = 1000       # new inclusion rule: keep subjects with >= 1000 total trials
USE_CHOICE_TRIALS_FOR_QC = True   # True = threshold on non-no-go trials; False = all trials

# 1) discover subjects that have any left-camera DLC
if SUBJECT_ALLOWLIST is not None:
    subjects = list(SUBJECT_ALLOWLIST)
else:
    dlc_eids = one.search(datasets=['_ibl_leftCamera.dlc.pqt'], query_type='remote')
    subjects = sorted({one.get_details(e)['subject'] for e in dlc_eids})

print(f'{len(subjects)} candidate subjects with left-camera DLC on the server')

# 2) build the QC availability table
qc_rows, subj_sessions, SEX = [], {}, {}
for s in tqdm(subjects, desc='QC sweep', unit='subject'):
    dlc = find_pupil_sessions(s)
    vid = find_video_sessions(s)
    subj_sessions[s] = dlc
    SEX[s] = get_sex(s)

    n_trials_total = 0
    n_choice_total = 0
    for eid in dlc:
        try:
            tr_i = load_session_trials(eid)
            n_trials_total += len(tr_i)
            n_choice_total += int((tr_i['choice'] != 0).sum())
        except Exception:
            pass

    qc_rows.append({
        'subject': s,
        'sex': SEX[s],
        'n_video': len(vid),
        'n_dlc': len(dlc),
        'n_trials_total': n_trials_total,
        'n_choice_total': n_choice_total,
    })

if len(qc_rows) == 0:
    raise RuntimeError(
        "qc_rows is empty: no subjects were discovered. "
        "Check ONE connection and remote dataset search."
    )

qc = pd.DataFrame(qc_rows).sort_values('subject').reset_index(drop=True)

print('\n=== VIDEO / DLC QC TABLE ===')
print(qc.to_string(index=False))

trial_col = 'n_choice_total' if USE_CHOICE_TRIALS_FOR_QC else 'n_trials_total'

# 3) inclusion rule -> CANDIDATES (the final dataset)
CANDIDATES = qc.loc[
    (qc['n_dlc'] >= MIN_DLC_SESSIONS) & (qc[trial_col] >= MIN_TOTAL_TRIALS),
    'subject'
].tolist()

print(
    f'\n{len(CANDIDATES)} animals pass QC '
    f'(n_dlc >= {MIN_DLC_SESSIONS}, {trial_col} >= {MIN_TOTAL_TRIALS}):'
)
print(CANDIDATES)



# ===== CELL 8 =====
demo_subj = 'CSHL045' if 'CSHL045' in CANDIDATES else CANDIDATES[0]
demo_eid  = subj_sessions[demo_subj][0]
tr = load_session_trials(demo_eid)
print('n_trials:', len(tr))
print('probabilityLeft:', dict(tr['probabilityLeft'].value_counts()))
tr.head()



# ===== CELL 9 =====
demo_subj = 'CSHL045' if 'CSHL045' in CANDIDATES else CANDIDATES[0]
demo_eid = subj_sessions[demo_subj][0]
tr = load_session_trials(demo_eid)
print('n_trials:', len(tr))
print('probabilityLeft:', dict(tr['probabilityLeft'].value_counts()))

tr = label_trial_epochs(tr)
print(dict(tr['epoch'].value_counts()))



# ===== CELL 10 =====
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

tr = label_trial_epochs(tr)
print(dict(tr['epoch'].value_counts()))



# ===== CELL 11 =====
def rightward_choice(df):
    # IBL choice == -1 -> mouse reported stimulus on the right
    return (df['choice'].to_numpy() == -1).astype(float)

def plot_psychometric_by_epoch(trials_df, ax=None):
    ax = ax or plt.gca()
    df = trials_df[trials_df['choice'] != 0].copy()
    df['rchoice'] = rightward_choice(df)
    for ep, col in [('unbiased','k'), ('stable','tab:blue'), ('transition','tab:orange')]:
        sub = df[df['epoch'] == ep]
        if not len(sub):
            continue
        g = sub.groupby('signed_contrast')['rchoice'].mean()
        ax.plot(g.index, g.values, '-o', ms=4, color=col, label=f'{ep} (n={len(sub)})')
    ax.axhline(.5, color='gray', ls=':'); ax.axvline(0, color='gray', ls=':')
    ax.set(xlabel='signed contrast (R+ / L-)', ylabel='P(rightward choice)',
           title='Psychometric by epoch'); ax.legend(fontsize=8)

def plot_accuracy_around_transitions(trials_df, window=(-5, 20), ax=None):
    ax = ax or plt.gca()
    df = trials_df.reset_index(drop=True)
    pL = df['probabilityLeft'].to_numpy()
    ch = np.zeros(len(df), bool); ch[1:] = (pL[1:] != pL[:-1]) & (pL[1:] != .5)
    idx = np.where(ch)[0]
    offs = np.arange(window[0], window[1]+1)
    mat = np.full((len(idx), len(offs)), np.nan)
    acc = df['accuracy'].to_numpy(dtype=float)
    for i, t0 in enumerate(idx):
        for j, o in enumerate(offs):
            k = t0 + o
            if 0 <= k < len(df):
                mat[i, j] = acc[k]
    m = np.nanmean(mat, axis=0)
    se = np.nanstd(mat, axis=0)/np.sqrt(np.sum(~np.isnan(mat), axis=0))
    ax.plot(offs, m, '-o', ms=3); ax.fill_between(offs, m-se, m+se, alpha=.3)
    ax.axvline(0, color='r', ls='--', label='block change')
    ax.set(xlabel='trial rel. to block change', ylabel='accuracy',
           title='Accuracy around biased-block transitions'); ax.legend(fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
plot_psychometric_by_epoch(tr, axes[0])
plot_accuracy_around_transitions(tr, ax=axes[1])
plt.tight_layout(); plt.show()



# ===== CELL 12 =====
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

# fit the demo animal across all its DLC sessions
demo_trials = [label_trial_epochs(load_session_trials(e)) for e in subj_sessions[demo_subj]]
model, demo_df, order = fit_animal_glmhmm(demo_trials, verbose=True)
print('\nGLM weights ordered [engaged, biased-left, biased-right] x [contrast, bias, prev_choice, prev_stim]:')
print(np.round(model.W[order], 2))
print('state labels (by bias-weight sign):', STATE_LABELS)
print('self-transition probs:', np.round(np.diag(model.A[np.ix_(order, order)]), 3))
cv = model.cv_
print(f'\nheld-out cross-validation ({cv["n_test_sess"]}/{cv["n_sess"]} sessions held out):')
print(f'  train log-lik/trial = {cv["train_ll"]:.4f}  (n={cv["n_train"]})')
print(f'  test  log-lik/trial = {cv["test_ll"]:.4f}  (n={cv["n_test"]})')
print('  (test close to train => the 3-state model generalizes, not overfit;')
print('   a chance model is ~-0.693 nats/trial, so higher = better than coin-flip.)')



# ===== CELL 13 =====
fig, ax = plt.subplots(figsize=(6, 4.5))
dd = demo_df[demo_df['choice'] != 0].copy(); dd['rchoice'] = rightward_choice(dd)
for k, col in zip(range(N_STATES), ['tab:green','tab:red','tab:purple']):
    sub = dd[dd['state'] == k]
    if len(sub) < 20:
        continue
    g = sub.groupby('signed_contrast')['rchoice'].mean()
    ax.plot(g.index, g.values, '-o', ms=4, color=col,
            label=f'{STATE_LABELS[k]}, n={len(sub)}')
ax.axhline(.5, color='gray', ls=':'); ax.axvline(0, color='gray', ls=':')
ax.set(xlabel='signed contrast', ylabel='P(rightward)', title=f'{demo_subj}: psychometric by inferred state')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()



# ===== CELL 14 =====
def occupancy_by_epoch(df, n_states=N_STATES):
    rows = []
    for ep in ['unbiased', 'transition', 'stable']:
        sub = df[df['epoch'] == ep]
        if not len(sub):
            continue
        row = dict(epoch=ep, n=len(sub))
        for k in range(n_states):
            row[f'occ_s{k}'] = np.mean(sub['state'] == k)
            m = sub.loc[sub['state'] == k, 'accuracy']
            row[f'acc_s{k}'] = m.mean() if len(m) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

occ = occupancy_by_epoch(demo_df)
print(occ.to_string(index=False))



# ===== CELL 15 =====
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

def zscore_within_animal(x):
    # Per-ANIMAL z-score (not per-session) -- used ONLY for the Spearman
    # correlation vs P(engaged), where scale-free within-animal comparison is
    # wanted. State LEVEL contrasts use the raw tonic/phasic values so that
    # between-state differences are NOT normalized away.
    x = np.asarray(x, float); m = np.nanmean(x); sd = np.nanstd(x)
    return (x - m)/sd if sd > 0 else x*np.nan



# ===== CELL 16 =====
tonic_pieces, phasic_pieces = [], []
for eid, d0 in zip(subj_sessions[demo_subj], demo_trials):
    d = d0[d0['choice'] != 0].reset_index(drop=True)
    diam, ct = load_pupil_diameter(eid)
    if diam is None or np.isnan(diam).mean() > NAN_GATE:
        tn = ph = np.full(len(d), np.nan)
    else:
        tn, ph = per_trial_pupil(diam, ct, d)
        # express TONIC as within-session deviation from the session median:
        # subtracting a per-session constant puts sessions on a common scale WITHOUT
        # centering each state to zero (unlike full z-scoring), so between-state
        # LEVEL differences survive. Phasic is already baseline-relative (px).
        tn = tn - np.nanmedian(tn)
    tonic_pieces.append(pd.Series(tn)); phasic_pieces.append(pd.Series(ph))

# demo_df rows are session-concatenated in the same order as demo_trials -> align directly
demo_df = demo_df.reset_index(drop=True)
demo_df['pupil_tonic']  = pd.concat(tonic_pieces,  ignore_index=True).values
demo_df['pupil_phasic'] = pd.concat(phasic_pieces, ignore_index=True).values

for feat in ['pupil_tonic', 'pupil_phasic']:
    d = demo_df.dropna(subset=[feat])
    print(f'\n===== {feat}  (n={len(d)} trials) =====')
    print(d.groupby('state')[feat].agg(['mean','std','count']).round(3))
    # descriptive only for one animal -- no across-state p-value (single subject,
    # pseudoreplicated). The population question is answered at the animal level
    # in Stage 5. Spearman below is a scale-free within-animal trend, not inference.
    rho, pr = spearmanr(zscore_within_animal(d[feat].values), d['p_state0'].values)
    print(f'  Spearman z({feat}) vs P(engaged): rho={rho:.3f} (within-animal trend, descriptive)')



# ===== CELL 17 =====
from scipy.ndimage import gaussian_filter

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
colors = ['tab:green', 'tab:red', 'tab:purple']

for row, feat in enumerate(['pupil_tonic', 'pupil_phasic']):
    d = demo_df.dropna(subset=[feat]).copy()
    states = sorted(d['state'].dropna().unique())

    vals = [d.loc[d['state'] == k, feat].to_numpy() for k in states]
    vp = ax[row, 0].violinplot(
        vals,
        positions=states,
        widths=0.8,
        showmeans=True,
        showmedians=False,
        showextrema=False
    )

    for body, c in zip(vp['bodies'], colors[:len(states)]):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.45)

    if 'cmeans' in vp:
        vp['cmeans'].set_color('k')
        vp['cmeans'].set_linewidth(1.5)

    means = d.groupby('state')[feat].mean().reindex(states)
    sems  = d.groupby('state')[feat].sem().reindex(states)
    ax[row, 0].errorbar(
        states, means.values, yerr=sems.values,
        fmt='o', color='k', capsize=4, lw=1.2, ms=4, zorder=3
    )

    ax[row, 0].axhline(0, color='0.75', lw=0.8, zorder=0)
    ax[row, 0].set(
        xlabel='latent state',
        ylabel='pupil (px)',
        title=f'{feat} by latent state'
    )
    ax[row, 0].set_xticks(states)

    x = d['p_state0'].to_numpy()
    y = d[feat].to_numpy()

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[60, 60],
        range=[[0, 1], [-5, 5]]
    )
    Hs = gaussian_filter(H, sigma=1.2)

    im = ax[row, 1].imshow(
        Hs.T,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='Blues'
    )

    ax[row, 1].axhline(0, color='0.75', lw=0.8, zorder=0)
    ax[row, 1].set(
        xlabel='P(engaged state)',
        ylabel='pupil z-score',
        title=f'{demo_subj}: {feat} vs P(engaged)',
        ylim=(-5, 5)
    )

    cb = fig.colorbar(im, ax=ax[row, 1])
    cb.set_label('smoothed trial density')

plt.tight_layout()
plt.show()



# ===== CELL 18 =====
import time

def estimate_subject_trials(subject, max_sessions=4):
    eids = subj_sessions.get(subject, [])[:max_sessions]
    out = dict(subject=subject, n_sessions=0, est_trials=0, est_choice_trials=0)
    if not eids:
        return out

    out['n_sessions'] = len(eids)
    for e in eids:
        try:
            tr = load_session_trials(e)
            out['est_trials'] += len(tr)
            out['est_choice_trials'] += int((tr['choice'] != 0).sum())
        except Exception:
            pass
    return out

work_rows = [estimate_subject_trials(s, max_sessions=4) for s in roster]
work = pd.DataFrame(work_rows).sort_values(
    ['est_choice_trials', 'n_sessions', 'subject'],
    ascending=[False, False, True]
).reset_index(drop=True)

print(work.to_string(index=False))

TOTAL_EST_TRIALS = int(work['est_choice_trials'].sum())
print(f'\nTotal estimated usable choice trials: {TOTAL_EST_TRIALS}')



# ===== CELL 19 =====
def run_animal(subject, max_sessions=4, verbose=False):
    eids = subj_sessions.get(subject, [])[:max_sessions]
    if not eids:
        return None
    sess = []
    for e in eids:
        try:
            sess.append((e, label_trial_epochs(load_session_trials(e))))
        except Exception as ex:
            print(f'  skip {subject} {e[:8]}: {ex}')
    if not sess:
        return None
    model, sdf, order = fit_animal_glmhmm([s[1] for s in sess])
    # attach pupil per session (rows align with the session-concatenated sdf)
    sdf = sdf.reset_index(drop=True)
    tonic_pieces, phasic_pieces = [], []
    for e, d0 in sess:
        d = d0[d0['choice'] != 0].reset_index(drop=True)
        try:
            diam, ct = load_pupil_diameter(e)
        except Exception as ex:
            # A single bad session (e.g. get_smooth_pupil_diameter raises when
            # raw pupil trace is >90% NaN) must not drop the whole ANIMAL -- only
            # this session's pupil columns become NaN below; its GLM-HMM/choice
            # data (already fit above) is untouched. Previously this exception
            # propagated out of run_animal and was only caught at the roster
            # loop, discarding every other good session for this animal too.
            print(f'  pupil load failed {subject} {e[:8]}: {ex}')
            diam, ct = None, None
        if diam is None or np.isnan(diam).mean() > NAN_GATE:
            tn = ph = np.full(len(d), np.nan)
        else:
            tn, ph = per_trial_pupil(diam, ct, d)
            tn = tn - np.nanmedian(tn)   # within-session deviation (preserves state levels)
        tonic_pieces.append(pd.Series(tn)); phasic_pieces.append(pd.Series(ph))
    sdf['pupil_tonic']  = pd.concat(tonic_pieces,  ignore_index=True).values
    sdf['pupil_phasic'] = pd.concat(phasic_pieces, ignore_index=True).values
    sdf['subject'] = subject; sdf['sex'] = SEX.get(subject)
    return dict(subject=subject, sex=SEX.get(subject), model=model, order=order,
                df=sdf, W=model.W[order])

# ---- ROSTER: sex-BALANCED sample, males matched to females by pupil-session count ----
# The IBL pupil cohort is male-skewed, so a 50/50 design is capped by the number
# of females available. We take ALL females, then for each female pick the unused
# male whose pupil-session count (n_dlc) is closest -- so the two sexes are matched
# on data quantity, not just count. Result: N_PER_SEX females + N_PER_SEX males.
# NOTE ON SAMPLE SIZE (honest): the earlier n=40/95%-power figure was invalid --
# it was derived from the 7-mouse PILOT using the FLUX metric, which turned out to
# be noise, and was then mis-applied to the tonic metric (a different, much smaller
# effect). Pilot-based effect sizes at n=7 are also unreliable (Kraemer 2006;
# Albers & Lakens 2018). At the animal level, for a small tonic effect (d~0.1-0.2)
# the required N is in the hundreds, not tens -- so 36 is an EXPLORATORY convenience
# sample, not a confirmatory design. The honest per-effect power is computed and
# printed in the dedicated power cell below.
qc_pass = qc[qc['n_dlc'] >= MIN_DLC_SESSIONS].copy()
females = qc_pass[qc_pass['sex'] == 'F'].sort_values('n_dlc', ascending=False)
males   = qc_pass[qc_pass['sex'] == 'M'].copy()
N_PER_SEX = min(len(females), len(males))
print(f'available: {len(females)} F, {len(males)} M  ->  balanced N_PER_SEX = {N_PER_SEX}')

# greedy nearest-n_dlc match: for each female (rarer sex), grab closest unused male
fem_roster = females.head(N_PER_SEX)
male_pool = males.set_index('subject')['n_dlc'].to_dict()
matched_males, match_log = [], []
for _, frow in fem_roster.iterrows():
    if not male_pool:
        break
    target = frow['n_dlc']
    best = min(male_pool, key=lambda m: (abs(male_pool[m] - target), m))
    matched_males.append(best)
    match_log.append((frow['subject'], int(target), best, int(male_pool[best])))
    del male_pool[best]

roster = fem_roster['subject'].tolist() + matched_males
TARGET_N = len(roster)
print(f'\nsex-balanced roster: {len(fem_roster)} F + {len(matched_males)} M = {TARGET_N} animals')
print('\nfemale -> matched male (by n_dlc):')
for fs, fn, ms, mn in match_log:
    print(f'  {fs:12s}(n_dlc={fn})  <->  {ms:12s}(n_dlc={mn})')
print('\nroster:', roster)
if N_PER_SEX < 20:
    print(f'\n[note] {N_PER_SEX}/sex ({TARGET_N} total) is the max even split this cohort '
          f'allows. Treat {TARGET_N} as an exploratory sample (see honest power cell).')



# ===== CELL 20 =====
work = qc.loc[
    qc['subject'].isin(CANDIDATES),
    ['subject', 'sex', 'n_dlc', 'n_video', 'n_trials_total', 'n_choice_total']
].copy()

work = work.rename(columns={
    'n_dlc': 'n_sessions',
    'n_choice_total': 'est_choice_trials'
})

work = work.sort_values(['sex', 'est_choice_trials', 'subject'], ascending=[True, False, True]).reset_index(drop=True)

# Optional: sex-balance to 56 F + 56 M if both exist in the current candidate set.
if (work['sex'].eq('F').sum() >= 56) and (work['sex'].eq('M').sum() >= 56):
    work = pd.concat([
        work[work['sex'] == 'F'].head(56),
        work[work['sex'] == 'M'].head(56),
    ], ignore_index=True).sort_values(['sex', 'est_choice_trials', 'subject'], ascending=[True, False, True]).reset_index(drop=True)

TOTAL_EST_TRIALS = int(work['est_choice_trials'].sum())
print(work[['subject', 'sex', 'n_sessions', 'est_choice_trials']].to_string(index=False))
print(f'TOTAL_EST_TRIALS = {TOTAL_EST_TRIALS}')



# ===== CELL 21 =====
print("Tonic clean trials:", len(tonic_trial_df))
print("Phasic clean trials:", len(phasic_trial_df))

print("\ntonic_df:")
print(tonic_df.shape)
print(tonic_df[["engaged", "biased_mean", "delta"]].describe())

print("\nphasic_df:")
print(phasic_df.shape)
print(phasic_df[["engaged", "biased_mean", "delta"]].describe())

assert len(tonic_trial_df) == 255_173
assert len(phasic_trial_df) == 253_405

assert np.isfinite(
    tonic_df[["engaged", "biased_mean"]].to_numpy()
).any()

assert np.isfinite(
    phasic_df[["engaged", "biased_mean"]].to_numpy()
).any()



# ===== CELL 22 =====
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_population_test(
    summary_df,
    feature_name,
    outlier_subjects=None,
):
    """
    Primary paired Wilcoxon analysis plus a sensitivity analysis
    excluding previously flagged animal-level delta outliers.
    """
    columns = [
        "subject",
        "engaged",
        "biased_mean",
        "delta",
    ]

    data = (
        summary_df[columns]
        .dropna(
            subset=[
                "engaged",
                "biased_mean",
                "delta",
            ]
        )
        .copy()
    )

    def run_test(test_data, label):
        differences = (
            test_data["engaged"]
            - test_data["biased_mean"]
        ).to_numpy(dtype=float)

        statistic, p_value = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
        )

        difference_sd = differences.std(ddof=1)

        dz = (
            differences.mean() / difference_sd
            if difference_sd > 0
            else np.nan
        )

        print(f"\n{feature_name}: {label}")
        print(f"n animals: {len(differences)}")
        print(
            f"engaged mean: "
            f"{test_data['engaged'].mean():.4f}"
        )
        print(
            f"biased mean:  "
            f"{test_data['biased_mean'].mean():.4f}"
        )
        print(
            f"mean delta:   "
            f"{differences.mean():.4f}"
        )
        print(
            f"median delta: "
            f"{np.median(differences):.4f}"
        )
        print(
            f"Wilcoxon W={statistic:.1f}, "
            f"p={p_value:.6g}"
        )
        print(f"paired dz={dz:.3f}")

    # Primary analysis: retain every QC-passed animal.
    run_test(
        data,
        label="all QC-passed animals",
    )

    # Sensitivity analysis only.
    if outlier_subjects:
        sensitivity_data = data.loc[
            ~data["subject"]
            .astype(str)
            .isin(outlier_subjects)
        ].copy()

        run_test(
            sensitivity_data,
            label=(
                "sensitivity analysis excluding "
                "animal-level delta outliers"
            ),
        )


paired_population_test(
    tonic_df,
    feature_name="Tonic pupil",
    outlier_subjects=tonic_outliers,
)

paired_population_test(
    phasic_df,
    feature_name="Phasic pupil",
    outlier_subjects=phasic_outliers,
)



# ===== CELL 23 =====
results, skipped = {}, []
done_trials = 0
done_animals = 0
t0 = time.time()

for s in work['subject']:
    est_trials = int(work.loc[work['subject'] == s, 'est_choice_trials'].iloc[0])
    n_sess = int(work.loc[work['subject'] == s, 'n_sessions'].iloc[0])

    elapsed = time.time() - t0
    trial_rate = done_trials / elapsed if elapsed > 0 else np.nan
    eta = (TOTAL_EST_TRIALS - done_trials) / trial_rate if trial_rate and np.isfinite(trial_rate) and trial_rate > 0 else np.nan

    print(
        f'--- {s} ---  '
        f'[{done_animals}/{len(work)} animals done]  '
        f'[{done_trials}/{TOTAL_EST_TRIALS} est trials done]  '
        f'next est={est_trials} trials, {n_sess} sess  '
        f'elapsed={elapsed:7.1f}s eta={eta:7.1f}s',
        flush=True
    )

    try:
        r = run_animal(s)
    except Exception as ex:
        r = None
        print(f' ERROR: {ex}', flush=True)

    done_animals += 1
    done_trials += est_trials

    if r is not None:
        results[s] = r
        actual_trials = len(r['df'])
        print(
            f' fit OK, actual={actual_trials} trials, engaged stim-weight={r["W"][0,0]:.2f}',
            flush=True
        )
    else:
        skipped.append(s)
        print(' skipped', flush=True)

print('\nanimals fitted:', list(results))
if skipped:
    print('skipped (no usable sessions):', skipped)

assert set(results) | set(skipped) == set(roster), 'roster/results mismatch'
print(f'accounted for {len(results)}+{len(skipped)} = {len(roster)} attempted animals')



# ===== CELL 24 =====
# =====================================================================
# PUPIL QC + REBUILD ALL DOWNSTREAM TONIC/PHASIC PLOTTING TABLES
#
# PLACE THIS CELL IMMEDIATELY AFTER:
#     trials_df = pd.concat(trial_frames, ...)
#     summary_df = pd.DataFrame(summary_rows)
#
# This replaces the old cell that starts with:
#     trials_df = trials_df.copy()
# =====================================================================

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Preserve the original flattened trial table
# ---------------------------------------------------------------------

pupil_trials_all = trials_df.copy()

# Remove the known bad animal BEFORE creating any summaries or plots.
# The old notebook removes SH015 in a later cell, after some plot tables
# have already been generated, which is too late.
pupil_trials_all = pupil_trials_all.loc[
    pupil_trials_all["subject"] != "SH015"
].copy()


# ---------------------------------------------------------------------
# 2. Robust z-score helper
# ---------------------------------------------------------------------

def robust_zscore(series):
    """
    Median/MAD robust z-score.

    Uses a standard-deviation fallback when MAD is zero. This matters for
    sessions or animals whose values are nearly constant.
    """
    x = pd.to_numeric(series, errors="coerce").astype(float)

    median_value = x.median(skipna=True)
    mad_value = (x - median_value).abs().median(skipna=True)

    robust_scale = 1.4826 * mad_value

    if (
        not np.isfinite(robust_scale)
        or robust_scale <= np.finfo(float).eps
    ):
        robust_scale = x.std(skipna=True, ddof=0)

    if (
        not np.isfinite(robust_scale)
        or robust_scale <= np.finfo(float).eps
    ):
        return pd.Series(
            np.zeros(len(x), dtype=float),
            index=x.index,
        )

    return (x - median_value) / robust_scale


# ---------------------------------------------------------------------
# 3. Choose the QC grouping level
# ---------------------------------------------------------------------

# Prefer QC within recording session when session metadata exists.
# With the notebook as currently written, it will probably fall back to
# subject-level QC because run_animal() does not yet retain the eid.
session_candidates = [
    "eid",
    "session_id",
    "session",
    "session_idx",
]

session_col = next(
    (col for col in session_candidates if col in pupil_trials_all.columns),
    None,
)

if session_col is None:
    pupil_qc_group_cols = ["subject"]
    print(
        "Pupil QC grouping: within subject. "
        "Add eid/session_idx in run_animal() for session-level QC."
    )
else:
    pupil_qc_group_cols = ["subject", session_col]
    print(
        f"Pupil QC grouping: within subject and {session_col}."
    )


# ---------------------------------------------------------------------
# 4. Calculate tonic and phasic QC independently
# ---------------------------------------------------------------------

TONIC_RZ_THRESHOLD = 5.0
PHASIC_RZ_THRESHOLD = 4.0

pupil_trials_all["pupil_tonic_rz"] = (
    pupil_trials_all
    .groupby(pupil_qc_group_cols, dropna=False)["pupil_tonic"]
    .transform(robust_zscore)
)

pupil_trials_all["pupil_phasic_rz"] = (
    pupil_trials_all
    .groupby(pupil_qc_group_cols, dropna=False)["pupil_phasic"]
    .transform(robust_zscore)
)

# Important:
# Tonic and phasic get separate validity flags. A bad or missing phasic
# value should not remove an otherwise usable tonic trial, and vice versa.
pupil_trials_all["pupil_tonic_ok"] = (
    np.isfinite(pupil_trials_all["pupil_tonic"])
    & pupil_trials_all["pupil_tonic_rz"].abs().le(
        TONIC_RZ_THRESHOLD
    )
)

pupil_trials_all["pupil_phasic_ok"] = (
    np.isfinite(pupil_trials_all["pupil_phasic"])
    & pupil_trials_all["pupil_phasic_rz"].abs().le(
        PHASIC_RZ_THRESHOLD
    )
)


# ---------------------------------------------------------------------
# 5. Create metric-specific clean trial tables
# ---------------------------------------------------------------------

# Use this table for every tonic analysis.
tonic_trial_df = pupil_trials_all.loc[
    pupil_trials_all["pupil_tonic_ok"]
].copy()

# Use this table for every phasic analysis.
phasic_trial_df = pupil_trials_all.loc[
    pupil_trials_all["pupil_phasic_ok"]
].copy()

# Use only when an analysis genuinely requires both measurements.
pupil_filtered_trial_df = pupil_trials_all.loc[
    pupil_trials_all["pupil_tonic_ok"]
    & pupil_trials_all["pupil_phasic_ok"]
].copy()

excluded_tonic_trial_df = pupil_trials_all.loc[
    ~pupil_trials_all["pupil_tonic_ok"]
].copy()

excluded_phasic_trial_df = pupil_trials_all.loc[
    ~pupil_trials_all["pupil_phasic_ok"]
].copy()


# ---------------------------------------------------------------------
# 6. Rebuild per-animal × state tables from CLEAN TRIALS
# ---------------------------------------------------------------------

STATE_LABELS_FOR_PUPIL = [
    "engaged",
    "biased-left",
    "biased-right",
]


def build_per_animal_state(
    clean_trials,
    feature,
    min_trials_per_state=10,
):
    """
    Create one value per animal, sex, and latent state.

    A state mean is set to NaN if fewer than min_trials_per_state clean
    trials remain for that animal/state combination.
    """
    required_columns = {
        "subject",
        "sex",
        "state_label",
        feature,
    }

    missing_columns = required_columns.difference(
        clean_trials.columns
    )

    if missing_columns:
        raise KeyError(
            f"Missing columns for {feature}: "
            f"{sorted(missing_columns)}"
        )

    grouped = (
        clean_trials
        .dropna(subset=[feature, "state_label"])
        .groupby(
            ["subject", "sex", "state_label"],
            dropna=False,
        )[feature]
        .agg(
            value="mean",
            n_trials="size",
        )
        .reset_index()
    )

    grouped.loc[
        grouped["n_trials"] < min_trials_per_state,
        "value",
    ] = np.nan

    per_animal = grouped.pivot_table(
        index=["subject", "sex"],
        columns="state_label",
        values="value",
        aggfunc="first",
    )

    # Ensure the expected columns always exist.
    for state_label in STATE_LABELS_FOR_PUPIL:
        if state_label not in per_animal.columns:
            per_animal[state_label] = np.nan

    return (
        per_animal[
            STATE_LABELS_FOR_PUPIL
        ]
        .sort_index()
    )


pa_tonic = build_per_animal_state(
    tonic_trial_df,
    feature="pupil_tonic",
    min_trials_per_state=10,
)

pa_phasic = build_per_animal_state(
    phasic_trial_df,
    feature="pupil_phasic",
    min_trials_per_state=10,
)


# ---------------------------------------------------------------------
# 7. Build tables expected by plot_summary()
# ---------------------------------------------------------------------

def build_paired_summary(per_animal):
    """
    Convert the state table into the format used by plot_summary():

        subject
        sex
        engaged
        biased-left
        biased-right
        biased_mean
        delta
    """
    summary = per_animal.reset_index().copy()

    biased_columns = [
        col
        for col in ["biased-left", "biased-right"]
        if col in summary.columns
    ]

    summary["biased_mean"] = summary[
        biased_columns
    ].mean(
        axis=1,
        skipna=True,
    )

    summary["delta"] = (
        summary["engaged"]
        - summary["biased_mean"]
    )

    return summary


tonic_df = build_paired_summary(pa_tonic)
phasic_df = build_paired_summary(pa_phasic)


# ---------------------------------------------------------------------
# 8. Flag animal-level effect outliers for display/sensitivity analysis
# ---------------------------------------------------------------------

def flag_animal_delta_outliers(
    summary_df,
    threshold=3.5,
):
    """
    Flag animals whose engaged-minus-biased effect is unusual.

    These animals are flagged rather than silently deleted. The existing
    plot_summary() function will draw them as red open circles.
    """
    delta_rz = robust_zscore(summary_df["delta"])

    return set(
        summary_df.loc[
            delta_rz.abs() > threshold,
            "subject",
        ].astype(str)
    )


tonic_outliers = flag_animal_delta_outliers(
    tonic_df,
    threshold=3.5,
)

phasic_outliers = flag_animal_delta_outliers(
    phasic_df,
    threshold=3.5,
)


# ---------------------------------------------------------------------
# 9. QC report
# ---------------------------------------------------------------------

def make_pupil_qc_report(
    all_trials,
    feature,
):
    ok_column = f"{feature}_ok"
    rz_column = f"{feature}_rz"

    report = (
        all_trials
        .groupby("subject", dropna=False)
        .apply(
            lambda sub: pd.Series(
                {
                    "n_total": len(sub),
                    "n_finite": int(
                        np.isfinite(sub[feature]).sum()
                    ),
                    "n_kept": int(sub[ok_column].sum()),
                    "n_excluded": int(
                        (~sub[ok_column]).sum()
                    ),
                    "fraction_excluded": (
                        (~sub[ok_column]).mean()
                    ),
                    "raw_min": sub[feature].min(
                        skipna=True
                    ),
                    "raw_max": sub[feature].max(
                        skipna=True
                    ),
                    "max_abs_robust_z": (
                        sub[rz_column]
                        .abs()
                        .max(skipna=True)
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    report.insert(1, "feature", feature)

    return report


tonic_qc_report = make_pupil_qc_report(
    pupil_trials_all,
    "pupil_tonic",
)

phasic_qc_report = make_pupil_qc_report(
    pupil_trials_all,
    "pupil_phasic",
)

pupil_qc_report = (
    pd.concat(
        [
            tonic_qc_report,
            phasic_qc_report,
        ],
        ignore_index=True,
    )
    .sort_values(
        [
            "fraction_excluded",
            "n_excluded",
        ],
        ascending=False,
    )
)


# ---------------------------------------------------------------------
# 10. Print enough information to verify the repair
# ---------------------------------------------------------------------

print("\n--- Pupil QC totals ---")
print(
    f"Tonic:  kept {len(tonic_trial_df):,} / "
    f"{len(pupil_trials_all):,} trials; "
    f"excluded {len(excluded_tonic_trial_df):,}"
)

print(
    f"Phasic: kept {len(phasic_trial_df):,} / "
    f"{len(pupil_trials_all):,} trials; "
    f"excluded {len(excluded_phasic_trial_df):,}"
)

print(
    f"Both valid: {len(pupil_filtered_trial_df):,} / "
    f"{len(pupil_trials_all):,} trials"
)

print("\nAnimal-level delta outliers:")
print("tonic:", sorted(tonic_outliers))
print("phasic:", sorted(phasic_outliers))

print("\nHighest-exclusion animal/feature combinations:")
print(
    pupil_qc_report
    .head(20)
    .to_string(index=False)
)

print("\nLargest excluded tonic values:")
print(
    excluded_tonic_trial_df[
        [
            "subject",
            "pupil_tonic",
            "pupil_tonic_rz",
            "state_label",
        ]
    ]
    .sort_values(
        "pupil_tonic_rz",
        key=lambda x: x.abs(),
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)

print("\nLargest excluded phasic values:")
print(
    excluded_phasic_trial_df[
        [
            "subject",
            "pupil_phasic",
            "pupil_phasic_rz",
            "state_label",
        ]
    ]
    .sort_values(
        "pupil_phasic_rz",
        key=lambda x: x.abs(),
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)



# ===== CELL 25 =====
# --- PATCH: flatten `results` into trial-level and animal-level tables ---
trial_frames = []
summary_rows = []

for subj, r in results.items():
    df_i = r['df'].copy()
    df_i['subject'] = subj
    df_i['sex'] = r.get('sex')
    trial_frames.append(df_i)

    W = r['W']  # already engagement-ordered (order applied in run_animal)
    row = {'subject': subj, 'sex': r.get('sex'), 'n_trials': len(df_i)}
    for k, label in enumerate(STATE_LABELS):
        for j, feat in enumerate(['stim', 'bias', 'prev_choice', 'prev_stim']):
            row[f'{label}_{feat}'] = W[k, j]
    summary_rows.append(row)

trials_df = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
summary_df = pd.DataFrame(summary_rows)

print(f'trials_df: {trials_df.shape}, summary_df: {summary_df.shape}')
print(f'skipped ({len(skipped)}):', skipped)

trials_df.to_csv('trial_level_pupil_glmhmm.csv', index=False)
summary_df.to_csv('animal_level_glmhmm_weights.csv', index=False)

import os
trials_path = os.path.abspath('trial_level_pupil_glmhmm.csv')
summary_path = os.path.abspath('animal_level_glmhmm_weights.csv')
trials_df.to_csv(trials_path, index=False)
summary_df.to_csv(summary_path, index=False)
print(trials_path)
print(summary_path)



# ===== CELL 26 =====
def plot_summary(ax, df, title, exclude=None, clip=None, label_top_n=1):
    exclude = set() if exclude is None else set(exclude)
    main = df.loc[~df["subject"].isin(exclude)].copy()
    out = df.loc[df["subject"].isin(exclude)].copy()

    for _, row in main.iterrows():
        ax.plot([0, 1], [row["engaged"], row["biased_mean"]], color="0.86", lw=0.8, zorder=1)

    ax.scatter(np.zeros(len(main)), main["engaged"], s=14, color="#4C78A8", alpha=0.75, zorder=2)
    ax.scatter(np.ones(len(main)), main["biased_mean"], s=14, color="#F58518", alpha=0.75, zorder=2)

    if len(out):
        ax.scatter(np.zeros(len(out)), out["engaged"], s=48, facecolors="none", edgecolors="crimson", lw=1.4, zorder=3)
        ax.scatter(np.ones(len(out)), out["biased_mean"], s=48, facecolors="none", edgecolors="crimson", lw=1.4, zorder=3)

        if label_top_n > 0:
            out = out.assign(abs_delta=out["delta"].abs()).sort_values("abs_delta", ascending=False).head(label_top_n)
            for _, row in out.iterrows():
                ax.annotate(row["subject"], (1, row["biased_mean"]), xytext=(4, 3),
                            textcoords="offset points", fontsize=7, color="crimson")

    ax.set_xticks([0, 1], ["engaged", "biased"])
    ax.set_title(title)
    ax.axhline(0, color="0.9", lw=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if clip is not None:
        ax.set_ylim(*clip)

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

plot_summary(axes[0, 0], tonic_df, "pupil_tonic (full)", exclude=tonic_outliers, label_top_n=4)
plot_summary(axes[0, 1], tonic_df, "pupil_tonic (clipped)", exclude=tonic_outliers, clip=(-3, 3), label_top_n=1)

plot_summary(axes[1, 0], phasic_df, "pupil_phasic (full)", exclude=phasic_outliers, label_top_n=4)
plot_summary(axes[1, 1], phasic_df, "pupil_phasic (clipped)", exclude=phasic_outliers, clip=(-3, 3), label_top_n=1)

plt.show()



# ===== CELL 27 =====
from scipy.stats import norm

def animal_level_power(pa, feat):
    # Effect = engaged vs mean-of-biased, PAIRED within animal (one pair per mouse).
    eng = pa['engaged']
    biased = pa[[c for c in pa.columns if c.startswith('biased')]].mean(axis=1)
    valid = eng.notna() & biased.notna()
    diff = (eng[valid] - biased[valid]).values
    n = len(diff)
    if n < 2 or np.nanstd(diff, ddof=1) == 0:
        print(f'{feat}: too few animals or zero variance (n={n})'); return
    dz = np.nanmean(diff) / np.nanstd(diff, ddof=1)      # paired Cohen's dz
    # N for 80% power, two-sided alpha=0.05, paired t (normal approx)
    za, zb = norm.ppf(1-0.05/2), norm.ppf(0.80)
    n_need = np.inf if dz == 0 else ((za+zb)/abs(dz))**2 + 1
    # achieved power at current n
    ncp = abs(dz)*np.sqrt(n)
    ach = norm.cdf(ncp - za) + norm.cdf(-ncp - za)
    print(f'{feat}:  observed paired dz={dz:+.3f} (n={n} mice)')
    print(f'    N needed for 80% power: {np.ceil(n_need):.0f} mice' if np.isfinite(n_need)
          else '    N needed for 80% power: effectively unbounded (dz~0)')
    print(f'    achieved power at n={n}: {ach*100:.0f}%')

for pa, feat in [(pa_tonic,'pupil_tonic'), (pa_phasic,'pupil_phasic')]:
    animal_level_power(pa, feat)



# ===== CELL 28 =====
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for axi, (pa, name) in zip(ax, [(pa_tonic,'pupil_tonic'), (pa_phasic,'pupil_phasic')]):
    eng = pa['engaged']
    biased = pa[[c for c in pa.columns if c.startswith('biased')]].mean(axis=1)
    valid = eng.notna() & biased.notna()
    # index is (subject, sex)
    for idx in pa.index[valid.values]:
        sx = idx[1] if isinstance(idx, tuple) and len(idx) > 1 else ''
        axi.plot([0,1], [eng.loc[idx], biased.loc[idx]], '-o',
                 color=('tab:blue' if sx=='M' else 'tab:orange'), alpha=.7)
    axi.set(xticks=[0,1], xticklabels=['engaged','biased'], ylabel=f'{name} (px)',
            title=name); axi.axhline(0, color='k', lw=.5, ls=':')
ax[0].plot([],[],'-o',color='tab:blue',label='M'); ax[0].plot([],[],'-o',color='tab:orange',label='F')
ax[0].legend(fontsize=8, title='sex')
plt.tight_layout(); plt.show()



# ===== CELL 29 =====
occ_rows = []
for s, r in results.items():
    o = occupancy_by_epoch(r['df'])
    o['subject'] = s; o['sex'] = r['sex']
    occ_rows.append(o)
occ_all = pd.concat(occ_rows, ignore_index=True)
print(occ_all[['subject','sex','epoch','n','occ_s0','acc_s0']].round(3).to_string(index=False))

# engaged occupancy: transition vs stable, per animal
piv = occ_all.pivot_table(index='subject', columns='epoch', values='occ_s0')
if {'transition','stable'}.issubset(piv.columns):
    both = piv.dropna(subset=['transition','stable'])
    if len(both) >= 3:
        diff_occ = (both['transition'] - both['stable']).values
        stat, p = wilcoxon(diff_occ)
        print(f'\nengaged-occupancy transition vs stable (Wilcoxon signed-rank): W={stat:.1f}, p={p:.3f}')



# ===== CELL 30 =====
summ = all_df.groupby(['subject','sex']).agg(
    pupil_tonic=('pupil_tonic','mean'),
    pupil_phasic=('pupil_phasic','mean'),
    engaged_frac=('state', lambda x: np.mean(x==0)),
).reset_index()
print(summ.round(3).to_string(index=False))

for metric in ['pupil_tonic','pupil_phasic','engaged_frac']:
    m = summ[summ['sex']=='M'][metric].dropna()
    f = summ[summ['sex']=='F'][metric].dropna()
    if len(m) >= 2 and len(f) >= 2:
        U, p = mannwhitneyu(m, f)
        print(f'{metric}: M={m.mean():.3f} (n={len(m)}) vs F={f.mean():.3f} (n={len(f)}), '
              f'Mann-Whitney U={U:.1f}, p={p:.3f}')
    else:
        print(f'{metric}: not enough animals per sex for a test (M={len(m)}, F={len(f)})')



# ===== CELL 31 =====
# Grouped violins: for each latent state, one violin per sex (M vs F).
# Two panels side by side: pupil TONIC (left, baseline arousal level) and
# pupil PHASIC (right, evoked change). Each point = one animal's mean value in
# that state (px); violin = across-animal distribution.
STATES = [c for c in ['engaged','biased-left','biased-right'] if c in pa_tonic.columns]
SEXES  = ['M','F']
SEX_COLOR = {'M':'tab:blue','F':'tab:red'}

def long_by_state_sex(pa):
    '''per-animal-by-state table (index=(subject,sex)) -> long df: state, sex, val.'''
    df = pa.reset_index()  # columns: subject, sex, <state cols>
    m = df.melt(id_vars=['subject','sex'], value_vars=STATES,
                var_name='state', value_name='val').dropna(subset=['val'])
    return m[m['sex'].isin(SEXES)]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
for ax, (pa, lab) in zip(axes, [(pa_tonic,'pupil_tonic (px)'), (pa_phasic,'pupil_phasic (px)')]):
    m = long_by_state_sex(pa)
    width = 0.34
    for si, st in enumerate(STATES):
        for k, sx in enumerate(SEXES):
            vals = m[(m['state']==st) & (m['sex']==sx)]['val'].values
            if len(vals) < 2:
                continue
            pos = si + (k - 0.5) * width
            vp = ax.violinplot(vals, positions=[pos], widths=width*0.9,
                               showmeans=True, showextrema=False)
            for body in vp['bodies']:
                body.set_facecolor(SEX_COLOR[sx]); body.set_alpha(0.45)
            if 'cmeans' in vp:
                vp['cmeans'].set_color(SEX_COLOR[sx])
            # jittered per-animal points
            jit = pos + (np.random.rand(len(vals)) - 0.5) * width * 0.5
            ax.scatter(jit, vals, s=12, color=SEX_COLOR[sx], alpha=0.7, zorder=3,
                       edgecolor='white', linewidth=0.3)
    ax.axhline(0, color='0.7', lw=0.8, zorder=0)
    ax.set_xticks(range(len(STATES)))
    ax.set_xticklabels(STATES, rotation=15)
    ax.set(xlabel='latent state', ylabel=lab, title=f'Per-state {lab} by sex')
# shared legend
handles = [plt.Line2D([],[],marker='o',ls='',color=SEX_COLOR[s],
                      label=f'{s} (n={pa_tonic.reset_index()["sex"].eq(s).sum()})') for s in SEXES]
axes[0].legend(handles=handles, fontsize=8, title='sex')
plt.tight_layout(); plt.show()



# ===== CELL 32 =====
def occupancy_by_pL(df, n_states=N_STATES):
    rows = []
    for pL in sorted(df['probabilityLeft'].dropna().unique()):
        sub = df[df['probabilityLeft'] == pL]
        if not len(sub):
            continue
        row = dict(probabilityLeft=pL, n=len(sub))
        for k in range(n_states):
            row[f'occ_s{k}'] = np.mean(sub['state'] == k)
        rows.append(row)
    return pd.DataFrame(rows)

pL_rows = []
for subject, res in results.items():
    if res is None:
        continue
    bo = occupancy_by_pL(res['df'])
    bo['subject'] = subject
    bo['sex'] = res['sex']
    pL_rows.append(bo)

pL_occ = pd.concat(pL_rows, ignore_index=True)
print(pL_occ.to_string(index=False))

from scipy.stats import wilcoxon

piv_left  = pL_occ.pivot_table(index='subject', columns='probabilityLeft', values='occ_s1')  # biased-left occupancy
piv_right = pL_occ.pivot_table(index='subject', columns='probabilityLeft', values='occ_s2')  # biased-right occupancy

# does biased-left occupancy rise when probabilityLeft=0.2 vs 0.8?
if 0.2 in piv_left.columns and 0.8 in piv_left.columns:
    d = (piv_left[0.2] - piv_left[0.8]).dropna()
    stat, p = wilcoxon(d)
    print(f'biased-left occupancy, pL=0.2 vs pL=0.8: paired Wilcoxon W={stat:.1f}, '
          f'p={p:.4f}, mean diff={d.mean():.3f}, n={len(d)} mice')

if 0.2 in piv_right.columns and 0.8 in piv_right.columns:
    d2 = (piv_right[0.8] - piv_right[0.2]).dropna()
    stat2, p2 = wilcoxon(d2)
    print(f'biased-right occupancy, pL=0.8 vs pL=0.2: paired Wilcoxon W={stat2:.1f}, '
          f'p={p2:.4f}, mean diff={d2.mean():.3f}, n={len(d2)} mice')



# ===== CELL 33 =====
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import os

pls = [0.2, 0.5, 0.8]

piv0 = pL_occ.pivot_table(index='subject', columns='probabilityLeft', values='occ_s0')
piv1 = pL_occ.pivot_table(index='subject', columns='probabilityLeft', values='occ_s1')
piv2 = pL_occ.pivot_table(index='subject', columns='probabilityLeft', values='occ_s2')

state_pivs = {'engaged': piv0, 'biased-left': piv1, 'biased-right': piv2}
colors = {'engaged': 'tab:blue', 'biased-left': 'tab:red', 'biased-right': 'tab:green'}

diff_left  = (piv1[0.2] - piv1[0.8]).dropna()
diff_right = (piv2[0.8] - piv2[0.2]).dropna()
w_left,  p_left  = wilcoxon(diff_left)
w_right, p_right = wilcoxon(diff_right)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# --- left panel: mean occupancy +/- SEM per state across probabilityLeft ---
ax = axes[0]
for label, piv in state_pivs.items():
    m = piv[pls].mean()
    se = piv[pls].sem()
    ax.errorbar(pls, m.values, yerr=se.values, marker='o', ms=8, lw=2.5,
                color=colors[label], label=label, capsize=4)
ax.set_xlabel('P(stim left)')
ax.set_ylabel('Mean occupancy')
ax.set_xticks(pls)
ax.set_title('Mean occupancy by block bias')
ax.legend(fontsize=9)

# --- right panel: paired per-animal occupancy shift ---
ax2 = axes[1]
bp = ax2.boxplot([diff_left.values, diff_right.values],
                  label=['biased-left\n(pL 0.2 - 0.8)', 'biased-right\n(pL 0.8 - 0.2)'],
                  patch_artist=True, widths=0.5)
for patch, label in zip(bp['boxes'], ['biased-left', 'biased-right']):
    patch.set_facecolor(colors[label])
    patch.set_alpha(0.4)

rng = np.random.default_rng(0)
for i, vals in enumerate([diff_left.values, diff_right.values], start=1):
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax2.scatter(np.full(len(vals), i) + jitter, vals, color='k', alpha=0.6, s=20, zorder=3)

ax2.axhline(0, color='gray', ls='--', lw=1)
ax2.set_ylabel('Occupancy diff (paired)')
ax2.set_title(f'biased-left p={p_left:.4f}, biased-right p={p_right:.4f}')

fig.suptitle(f'State occupancy tracks block-bias direction (n={len(diff_left)} mice)', y=1.02)
plt.tight_layout()
os.makedirs('output', exist_ok=True)
plt.savefig('output/occupancy_by_pL_summary.png', dpi=150, bbox_inches='tight')
plt.show()



# ===== CELL 34 =====
trial_counts = {s: len(res['df']) for s, res in results.items() if res is not None}
low_n = {s: n for s, n in trial_counts.items() if n < 1000}
print(f'{len(low_n)} of {len(trial_counts)} animals below 1000 trials:')
print(low_n)



# ===== CELL 35 =====
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

def build_per_animal(results, feat='pupil_tonic'):
    rows = []
    for subject, res in results.items():
        if res is None:
            continue
        d = res['df'].dropna(subset=[feat])
        row = {'subject': subject, 'sex': res['sex']}
        for label in STATE_LABELS:
            sub = d.loc[d['state_label'] == label, feat]
            row[label] = sub.mean() if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('subject')

per_animal = build_per_animal(results, feat='pupil_tonic')
CANDIDATES_1000 = [s for s, n in trial_counts.items() if n >= 1000]

eng = per_animal.loc[CANDIDATES_1000, 'engaged']
biased = per_animal.loc[CANDIDATES_1000, [c for c in per_animal.columns if c.startswith('biased')]].mean(axis=1)
valid = eng.notna() & biased.notna()
diff = (eng[valid] - biased[valid]).values
stat, p = wilcoxon(diff)
dz = diff.mean()/diff.std(ddof=1)
print(f'[n={valid.sum()}] tonic engaged-biased (filtered): W={stat:.1f}, p={p:.4f}, dz={dz:.3f}')



# ===== CELL 36 =====
def build_occ_all(results):
    rows = []
    for subject, res in results.items():
        if res is None:
            continue
        occ = occupancy_by_epoch(res['df'])
        occ['subject'] = subject
        rows.append(occ)
    return pd.concat(rows, ignore_index=True)

occ_all = build_occ_all(results)

piv = occ_all[occ_all['subject'].isin(CANDIDATES_1000)].pivot_table(
    index='subject', columns='epoch', values='occ_s0')
both = piv.dropna(subset=['transition', 'stable'])
diff_occ = (both['transition'] - both['stable']).values
stat2, p2 = wilcoxon(diff_occ)
dz2 = diff_occ.mean()/diff_occ.std(ddof=1)
print(f'[n={len(both)}] transition vs stable (filtered): W={stat2:.1f}, p={p2:.4f}, dz={dz2:.3f}')



# ===== CELL 37 =====
CANDIDATES_1000 = [s for s, n in trial_counts.items() if n >= 1000]

# tonic pupil, engaged vs biased — filtered to n=28
eng = per_animal.loc[CANDIDATES_1000, 'engaged']
biased = per_animal.loc[CANDIDATES_1000, [c for c in per_animal.columns if c.startswith('biased')]].mean(axis=1)
valid = eng.notna() & biased.notna()
diff = (eng[valid] - biased[valid]).values
stat, p = wilcoxon(diff)
dz = diff.mean()/diff.std(ddof=1)
print(f'[n={valid.sum()}] tonic engaged-biased (filtered): W={stat:.1f}, p={p:.4f}, dz={dz:.3f}')

# occupancy transition vs stable — filtered to n=28
piv = occ_all[occ_all['subject'].isin(CANDIDATES_1000)].pivot_table(
    index='subject', columns='epoch', values='occ_s0')
both = piv.dropna(subset=['transition', 'stable'])
diff_occ = (both['transition'] - both['stable']).values
stat2, p2 = wilcoxon(diff_occ)
dz2 = diff_occ.mean()/diff_occ.std(ddof=1)
print(f'[n={len(both)}] transition vs stable (filtered): W={stat2:.1f}, p={p2:.4f}, dz={dz2:.3f}')



# ===== CELL 38 =====
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t, nct

def power_paired(dz, n, alpha=0.05):
    df = n - 1
    ncp = dz * np.sqrt(n)
    tcrit = t.ppf(1 - alpha/2, df)
    return 1 - nct.cdf(tcrit, df, ncp) + nct.cdf(-tcrit, df, ncp)

def n_for_power(dz, power=0.8, alpha=0.05):
    for n in range(3, 5000):
        if power_paired(abs(dz), n, alpha) >= power:
            return n
    return None

def transition_sensitivity_check(results, candidates, n_trans_list=(10, 15, 20, 25)):
    out = []
    for ntrans in n_trans_list:
        occ_rows = []
        for subject in candidates:
            res = results.get(subject)
            if res is None:
                continue
            df = res['df']
            if 'eid' in df.columns:
                eids_used = df['eid'].drop_duplicates().tolist()
                relabeled = []
                for e in eids_used:
                    d = label_trial_epochs(load_session_trials(e), n_transition_trials=ntrans)
                    relabeled.append(d[d['choice'] != 0].reset_index(drop=True))
                merged = pd.concat(relabeled, ignore_index=True)
                if len(merged) != len(df):
                    print(f'  [SKIP] {subject}: {len(merged)} vs {len(df)} rows')
                    continue
                merged['state'] = df['state'].values
            else:
                # no eid column yet -- relabel epochs directly on stored df using
                # probabilityLeft, which is already present per trial
                merged = df.copy()
                pL = merged['probabilityLeft'].to_numpy()
                epoch = np.array(['stable'] * len(merged), dtype=object)
                epoch[pL == 0.5] = 'unbiased'
                changed = np.zeros(len(merged), bool)
                changed[1:] = pL[1:] != pL[:-1]
                for idx in np.where(changed)[0]:
                    if pL[idx] == 0.5:
                        continue
                    for j in range(idx, min(idx + ntrans, len(merged))):
                        if epoch[j] == 'stable':
                            epoch[j] = 'transition'
                merged['epoch'] = epoch

            occ = occupancy_by_epoch(merged)
            piv = occ.set_index('epoch')['occ_s0']
            if {'transition', 'stable'}.issubset(piv.index):
                occ_rows.append(dict(subject=subject, diff=piv['transition'] - piv['stable']))

        dd = pd.DataFrame(occ_rows).dropna()
        if len(dd) < 3:
            out.append(dict(n_transition_trials=ntrans, dz=np.nan, p=np.nan, n=len(dd),
                            power=np.nan, n_needed=np.nan))
            continue
        stat, p = wilcoxon(dd['diff'])
        dz = dd['diff'].mean() / dd['diff'].std(ddof=1)
        pw = power_paired(abs(dz), len(dd))
        nn = n_for_power(dz)
        out.append(dict(n_transition_trials=ntrans, dz=dz, p=p, n=len(dd),
                        power=pw, n_needed=nn,
                        n_pos=(dd['diff'] > 0).sum(), n_neg=(dd['diff'] < 0).sum()))
    return pd.DataFrame(out)

sens = transition_sensitivity_check(results, CANDIDATES_1000)
print(sens.to_string(index=False))

print('\ndz trajectory across window widths:', sens['dz'].round(3).tolist())
print('achieved power at n=28 stays below 80% at every window width:',
      (sens['power'] < 0.8).all())
print('=> a stable near-zero/low dz across all widths, combined with power < 80% '
      'throughout, means this sweep cannot distinguish "no transition effect" '
      'from "a true small effect masked by n=28" — report both the dz trajectory '
      'and this power ceiling, not a bare p-value.')



# ===== CELL 39 =====
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def relabel_epoch(df, n_transition_trials=20):
    d = df.copy()
    pL = d['probabilityLeft'].to_numpy()
    epoch = np.array(['stable'] * len(d), dtype=object)
    epoch[pL == 0.5] = 'unbiased'
    changed = np.zeros(len(d), bool)
    changed[1:] = pL[1:] != pL[:-1]
    for idx in np.where(changed)[0]:
        if pL[idx] == 0.5:
            continue
        for j in range(idx, min(idx + n_transition_trials, len(d))):
            if epoch[j] == 'stable':
                epoch[j] = 'transition'
    d['epoch'] = epoch
    return d

def tonic_pupil_by_epoch_and_state(results, candidates, feat='pupil_tonic', n_transition_trials=20):
    rows = []
    for subject in candidates:
        res = results.get(subject)
        if res is None:
            continue
        df = relabel_epoch(res['df'], n_transition_trials=n_transition_trials)
        d = df.dropna(subset=[feat])
        for ep in ['unbiased', 'transition', 'stable']:
            sub = d[d['epoch'] == ep]
            if not len(sub):
                continue
            eng = sub.loc[sub['state'] == 0, feat]
            biased = sub.loc[sub['state'] != 0, feat]
            if len(eng) >= 5 and len(biased) >= 5:
                rows.append(dict(subject=subject, sex=res['sex'], epoch=ep,
                                  eng_mean=eng.mean(), biased_mean=biased.mean(),
                                  n_eng=len(eng), n_biased=len(biased)))
    return pd.DataFrame(rows)

tp_epoch = tonic_pupil_by_epoch_and_state(results, CANDIDATES_1000, n_transition_trials=20)

summary = []
for ep in ['unbiased', 'transition', 'stable']:
    sub = tp_epoch[tp_epoch['epoch'] == ep]
    diff = (sub['eng_mean'] - sub['biased_mean']).dropna()
    if len(diff) >= 3:
        stat, p = wilcoxon(diff)
        dz = diff.mean() / diff.std(ddof=1)
        summary.append(dict(epoch=ep, W=stat, p=p, dz=dz, n=len(diff)))
        print(f'{ep:10s}: engaged-biased tonic pupil, W={stat:.1f}, p={p:.4f}, '
              f'dz={dz:.3f}, n={len(diff)} mice')

summary_df = pd.DataFrame(summary)
print('\nfor comparison, pooled (all epochs together) tonic result: p=0.028, dz=-0.387, n=28')



# ===== CELL 40 =====
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------
# Display terminology
#
# Task context:
#   unbiased   -> initial 50:50 stimulus-prior block
#   transition -> early trials after entering a biased-prior block
#   stable     -> later trials within a biased-prior block
#
# Latent decision strategy:
#   engaged
#   biased-left / biased-right, pooled here as "biased states"
# ---------------------------------------------------------------------

epochs = [
    "unbiased",
    "transition",
    "stable",
]

epoch_display = {
    "unbiased": "Unbiased block\n(50:50 prior)",
    "transition": "Early biased block",
    "stable": "Later biased block",
}

colors = {
    "engaged": "tab:blue",
    "biased": "tab:red",
}

rng = np.random.default_rng(0)

fig, ax = plt.subplots(
    figsize=(11, 6.8)
)

positions = []
data = []
labels = []

# Each entry:
# (engaged_position, biased_position, p, dz, n, group_type)
group_pairs = []

# Used to place task-block labels under each pair.
group_centers = []

pos = 1


# ---------------------------------------------------------------------
# Three task-block phase groups
# ---------------------------------------------------------------------

for epoch in epochs:
    sub = tp_epoch.loc[
        tp_epoch["epoch"] == epoch
    ].copy()

    # Paired values must come from the same animals.
    paired = sub[
        [
            "subject",
            "eng_mean",
            "biased_mean",
        ]
    ].dropna(
        subset=[
            "eng_mean",
            "biased_mean",
        ]
    )

    engaged_values = (
        paired["eng_mean"]
        .to_numpy(dtype=float)
    )

    biased_values = (
        paired["biased_mean"]
        .to_numpy(dtype=float)
    )

    pos_engaged = pos
    pos_biased = pos + 0.8

    data.extend(
        [
            engaged_values,
            biased_values,
        ]
    )

    positions.extend(
        [
            pos_engaged,
            pos_biased,
        ]
    )

    # Tick labels describe the latent GLM-HMM state.
    labels.extend(
        [
            "Engaged\nstate",
            "Biased states\n(L/R pooled)",
        ]
    )

    # Pair-level label describes the task block.
    group_centers.append(
        (
            (pos_engaged + pos_biased) / 2,
            epoch_display[epoch],
        )
    )

    difference = (
        engaged_values
        - biased_values
    )

    if len(difference) >= 3:
        statistic, p_value = wilcoxon(
            difference
        )

        difference_sd = difference.std(
            ddof=1
        )

        dz = (
            difference.mean()
            / difference_sd
            if difference_sd > 0
            else np.nan
        )

        group_pairs.append(
            (
                pos_engaged,
                pos_biased,
                p_value,
                dz,
                len(difference),
                "epoch",
            )
        )

    pos += 2.2


# ---------------------------------------------------------------------
# Fourth group: pooled across all task blocks
# ---------------------------------------------------------------------

pos += 1.0

engaged_pooled = (
    eng.loc[valid]
    .to_numpy(dtype=float)
)

biased_pooled = (
    biased.loc[valid]
    .to_numpy(dtype=float)
)

pos_engaged_pooled = pos
pos_biased_pooled = pos + 0.8

data.extend(
    [
        engaged_pooled,
        biased_pooled,
    ]
)

positions.extend(
    [
        pos_engaged_pooled,
        pos_biased_pooled,
    ]
)

labels.extend(
    [
        "Engaged\nstate",
        "Biased states\n(L/R pooled)",
    ]
)

group_centers.append(
    (
        (
            pos_engaged_pooled
            + pos_biased_pooled
        ) / 2,
        "All task blocks",
    )
)

pooled_difference = (
    engaged_pooled
    - biased_pooled
)

pool_statistic, pool_p = wilcoxon(
    pooled_difference
)

pool_sd = pooled_difference.std(
    ddof=1
)

pool_dz = (
    pooled_difference.mean()
    / pool_sd
    if pool_sd > 0
    else np.nan
)

group_pairs.append(
    (
        pos_engaged_pooled,
        pos_biased_pooled,
        pool_p,
        pool_dz,
        len(pooled_difference),
        "pooled",
    )
)


# ---------------------------------------------------------------------
# Draw violins
# ---------------------------------------------------------------------

parts = ax.violinplot(
    data,
    positions=positions,
    widths=0.7,
    showmeans=True,
    showextrema=False,
)

for index, body in enumerate(
    parts["bodies"]
):
    condition = (
        "engaged"
        if index % 2 == 0
        else "biased"
    )

    body.set_facecolor(
        colors[condition]
    )

    body.set_edgecolor(
        colors[condition]
    )

    body.set_alpha(0.35)


# ---------------------------------------------------------------------
# Draw individual animals
# ---------------------------------------------------------------------

for index, values in enumerate(data):
    jitter = rng.uniform(
        -0.12,
        0.12,
        size=len(values),
    )

    condition = (
        "engaged"
        if index % 2 == 0
        else "biased"
    )

    ax.scatter(
        np.full(
            len(values),
            positions[index],
        )
        + jitter,
        values,
        color=colors[condition],
        alpha=0.7,
        s=18,
        zorder=3,
        edgecolor="black",
        linewidth=0.3,
    )


# ---------------------------------------------------------------------
# Statistical annotations
# ---------------------------------------------------------------------

def significance_label(p_value):
    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "n.s."


y_max = max(
    np.max(values)
    for values in data
    if len(values) > 0
)

y_min = min(
    np.min(values)
    for values in data
    if len(values) > 0
)

y_range = y_max - y_min

if y_range == 0:
    y_range = 1.0

bracket_height = (
    y_range * 0.06
)


for (
    pos_engaged,
    pos_biased,
    p_value,
    dz,
    n_animals,
    group_type,
) in group_pairs:

    engaged_index = positions.index(
        pos_engaged
    )

    biased_index = positions.index(
        pos_biased
    )

    local_max = max(
        np.max(data[engaged_index]),
        np.max(data[biased_index]),
    )

    bracket_y = (
        local_max
        + bracket_height
    )

    line_width = (
        1.8
        if group_type == "pooled"
        else 1.2
    )

    ax.plot(
        [
            pos_engaged,
            pos_engaged,
            pos_biased,
            pos_biased,
        ],
        [
            bracket_y,
            bracket_y
            + bracket_height * 0.4,
            bracket_y
            + bracket_height * 0.4,
            bracket_y,
        ],
        color="black",
        linewidth=line_width,
    )

    annotation = (
        f"{significance_label(p_value)}\n"
        f"p={p_value:.3f}, "
        f"dz={dz:.2f}, "
        f"n={n_animals}"
    )

    ax.text(
        (
            pos_engaged
            + pos_biased
        ) / 2,
        bracket_y
        + bracket_height * 0.55,
        annotation,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=8.5,
        fontweight=(
            "bold"
            if group_type == "pooled"
            else "normal"
        ),
    )


# ---------------------------------------------------------------------
# Separate epoch-specific and pooled results
# ---------------------------------------------------------------------

divider_x = (
    positions[5]
    + pos_engaged_pooled
) / 2

ax.axvline(
    divider_x,
    color="gray",
    linestyle=":",
    linewidth=1,
    alpha=0.6,
)


# ---------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------

ax.set_ylim(
    y_min - y_range * 0.05,
    y_max + y_range * 0.42,
)

ax.set_xticks(
    positions
)

ax.set_xticklabels(
    labels,
    fontsize=9,
)

# Add task-block labels beneath each engaged/biased state pair.
for center_x, block_label in group_centers:
    ax.text(
        center_x,
        -0.18,
        block_label,
        transform=ax.get_xaxis_transform(),
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9.5,
        fontweight="bold",
        clip_on=False,
    )


ax.set_ylabel(
    "Per-animal mean tonic pupil"
)

ax.set_title(
    "Tonic pupil across task-block phases and latent decision states\n"
    "Paired Wilcoxon signed-rank tests; points represent mice"
)


# ---------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------

handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=colors["engaged"],
        markeredgecolor="none",
        markersize=8,
        label="Engaged GLM-HMM state",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=colors["biased"],
        markeredgecolor="none",
        markersize=8,
        label=(
            "Biased GLM-HMM states "
            "(left/right pooled)"
        ),
    ),
]

ax.legend(
    handles=handles,
    loc="upper right",
    fontsize=9,
)


# Leave room for the task-block labels beneath the x-axis.
plt.tight_layout(
    rect=[
        0,
        0.10,
        1,
        1,
    ]
)

plt.savefig(
    "output/tonic_pupil_by_epoch_and_pooled_violin.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()



# ===== CELL 41 =====
# Find the largest per-animal values entering the displayed figure.

epoch_extremes = (
    tp_epoch
    .assign(
        max_abs=lambda d: d[
            ["eng_mean", "biased_mean"]
        ].abs().max(axis=1)
    )
    .sort_values("max_abs", ascending=False)
)

print("Largest epoch-level values:")
print(
    epoch_extremes[
        [
            "subject",
            "epoch",
            "eng_mean",
            "biased_mean",
            "n_eng",
            "n_biased",
            "max_abs",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

pooled_debug = pd.DataFrame(
    {
        "subject": eng[valid].index,
        "engaged": eng[valid].to_numpy(),
        "biased": biased[valid].to_numpy(),
    }
)

pooled_debug["delta"] = (
    pooled_debug["engaged"]
    - pooled_debug["biased"]
)

pooled_debug["max_abs"] = (
    pooled_debug[
        ["engaged", "biased"]
    ]
    .abs()
    .max(axis=1)
)

print("\nLargest pooled values:")
print(
    pooled_debug
    .sort_values("max_abs", ascending=False)
    .head(15)
    .to_string(index=False)
)



# ===== CELL 42 =====
# ============================================================
# REMOVE KNOWN UNUSABLE PUPIL SUBJECT FROM CURRENT FIGURE DATA
# ============================================================

import numpy as np
from scipy.stats import wilcoxon

BAD_PUPIL_SUBJECTS = {"SH015"}


# Remove from per-epoch table
tp_epoch = tp_epoch.loc[
    ~tp_epoch["subject"]
    .astype(str)
    .isin(BAD_PUPIL_SUBJECTS)
].copy()


# Remove from pooled per-animal series
pooled_keep = (
    ~eng.index
    .astype(str)
    .isin(BAD_PUPIL_SUBJECTS)
)

eng = eng.loc[pooled_keep].copy()
biased = biased.loc[pooled_keep].copy()

valid = (
    eng.notna()
    & biased.notna()
)


# Recompute pooled statistics
pool_diff = (
    eng.loc[valid]
    - biased.loc[valid]
).to_numpy(dtype=float)

pool_w, pool_p = wilcoxon(
    pool_diff
)

pool_sd = pool_diff.std(ddof=1)

pool_dz = (
    pool_diff.mean() / pool_sd
    if pool_sd > 0
    else np.nan
)


# Sanity checks: fail loudly if corrupted values remain
epoch_max = (
    tp_epoch[
        ["eng_mean", "biased_mean"]
    ]
    .abs()
    .to_numpy()
    .max()
)

pooled_max = max(
    eng.abs().max(),
    biased.abs().max(),
)

assert epoch_max < 100, (
    f"Implausible epoch value remains: {epoch_max}"
)

assert pooled_max < 100, (
    f"Implausible pooled value remains: {pooled_max}"
)

print(
    f"SH015 removed from pupil figure.\n"
    f"Epoch rows remaining: {len(tp_epoch)}\n"
    f"Pooled animals remaining: {valid.sum()}\n"
    f"Pooled result: W={pool_w:.1f}, "
    f"p={pool_p:.6g}, dz={pool_dz:.3f}\n"
    f"Largest epoch value: {epoch_max:.3f}\n"
    f"Largest pooled value: {pooled_max:.3f}"
)



# ===== CELL 43 =====
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

# 1. Epoch labels partition every trial exactly once (no double-count, no drop)
mismatch = []
for subject in CANDIDATES_1000:
    res = results.get(subject)
    if res is None:
        continue
    df = relabel_epoch(res['df'], n_transition_trials=20)
    if len(df) != df['epoch'].value_counts().sum():
        mismatch.append(subject)
print("1. epoch coverage mismatch:", mismatch if mismatch else "none — all trials labeled exactly once")

# 2. Are the SAME mice driving the effect across epochs, or different mice each time?
# (If different mice dominate each epoch, "consistent dz across epochs" is coincidence,
# not a real within-animal stable phenomenon.)
piv_diff = tp_epoch.assign(diff=tp_epoch['eng_mean'] - tp_epoch['biased_mean']).pivot_table(
    index='subject', columns='epoch', values='diff')
corr_ut, p_ut = spearmanr(piv_diff['unbiased'], piv_diff['transition'], nan_policy='omit')
corr_ts, p_ts = spearmanr(piv_diff['transition'], piv_diff['stable'], nan_policy='omit')
corr_us, p_us = spearmanr(piv_diff['unbiased'], piv_diff['stable'], nan_policy='omit')
print(f"\n2. per-mouse diff correlation across epochs:")
print(f"   unbiased vs transition: rho={corr_ut:.3f}, p={p_ut:.3f}")
print(f"   transition vs stable:   rho={corr_ts:.3f}, p={p_ts:.3f}")
print(f"   unbiased vs stable:     rho={corr_us:.3f}, p={p_us:.3f}")

# 3. Sample-size sanity: n_eng and n_biased per animal per epoch shouldn't be
# pathologically small/imbalanced (e.g. n_eng=5000, n_biased=5 would make the
# biased-state mean unstable even if it clears the >=5 threshold)
imbalance = tp_epoch.assign(ratio=tp_epoch['n_eng'] / tp_epoch['n_biased'])
print("\n3. engaged:biased trial-count ratio per animal/epoch (max 5 shown):")
print(imbalance.sort_values('ratio', ascending=False)[['subject','epoch','n_eng','n_biased','ratio']].head())

# 4. Outlier check: does any single mouse's diff dominate the pooled Wilcoxon?
# Recompute pooled dz with each animal held out one at a time (leave-one-out).
pooled = piv_diff.mean(axis=1, skipna=True).dropna()  # crude per-animal mean diff across epochs
loo_dz = []
for excl in pooled.index:
    remaining = pooled.drop(excl)
    dz_loo = remaining.mean() / remaining.std(ddof=1)
    loo_dz.append((excl, dz_loo))
loo_df = pd.DataFrame(loo_dz, columns=['excluded', 'dz_without']).sort_values('dz_without')
print("\n4. leave-one-out dz (most sensitive exclusions at top/bottom):")
print(loo_df.head(3))
print(loo_df.tail(3))

# 5. Direction check: what fraction of mice actually show eng < biased (predicted
# direction) vs the opposite, in the pooled (all-epoch) diff used for the headline test?
pooled_diff = (eng[valid] - biased[valid])
n_pred = (pooled_diff < 0).sum()
n_opp = (pooled_diff > 0).sum()
print(f"\n5. pooled tonic diff direction: {n_pred}/{len(pooled_diff)} mice engaged<biased "
      f"(predicted), {n_opp}/{len(pooled_diff)} opposite")



# ===== CELL 44 =====
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

# filtered roster
trial_counts = {s: len(res['df']) for s, res in results.items() if res is not None}
CANDIDATES_1000 = sorted([s for s, n in trial_counts.items() if n >= 1000])

# pooled tonic pupil
eng = per_animal.loc[CANDIDATES_1000, 'engaged']
biased = per_animal.loc[CANDIDATES_1000, [c for c in per_animal.columns if c.startswith('biased')]].mean(axis=1)
valid = eng.notna() & biased.notna()
pool_diff = (eng[valid] - biased[valid]).values
pool_w, pool_p = wilcoxon(pool_diff)
pool_dz = pool_diff.mean() / pool_diff.std(ddof=1)
pool_n = int(valid.sum())

# leave-one-out pooled dz
loo_rows = []
idx = valid.index[valid]
for excl in idx:
    d = pool_diff[idx != excl]
    loo_rows.append({'excluded': excl, 'dz_without': d.mean() / d.std(ddof=1), 'n_without': len(d)})
loo_df = pd.DataFrame(loo_rows)

# epoch-stratified summaries
epoch_rows = []
for ep in ['unbiased', 'transition', 'stable']:
    sub = tp_epoch[tp_epoch['epoch'] == ep].copy()
    diff = (sub['eng_mean'] - sub['biased_mean']).dropna()
    if len(diff) >= 3:
        w, p = wilcoxon(diff)
        dz = diff.mean() / diff.std(ddof=1)
        epoch_rows.append({
            'analysis': f'tonic_{ep}',
            'n_mice': int(len(diff)),
            'W': float(w),
            'p': float(p),
            'dz': float(dz),
            'mean_diff': float(diff.mean()),
            'sd_diff': float(diff.std(ddof=1)),
            'median_n_eng': float(sub['n_eng'].median()),
            'median_n_biased': float(sub['n_biased'].median()),
            'n_mice_predicted_dir': int((diff < 0).sum()),
            'n_mice_opposite_dir': int((diff > 0).sum()),
        })

# correlations across epochs
piv_diff = tp_epoch.assign(diff=tp_epoch['eng_mean'] - tp_epoch['biased_mean']).pivot_table(
    index='subject', columns='epoch', values='diff'
)
cor_rows = []
for a, b, name in [('unbiased', 'transition', 'rho_unbiased_transition'),
                   ('transition', 'stable', 'rho_transition_stable'),
                   ('unbiased', 'stable', 'rho_unbiased_stable')]:
    sub = piv_diff[[a, b]].dropna()
    if len(sub) >= 3:
        rho, p = spearmanr(sub[a], sub[b])
        cor_rows.append({'analysis': name, 'n_pairs': int(len(sub)), 'rho': float(rho), 'p': float(p)})
cor_df = pd.DataFrame(cor_rows)

# main table
summary_rows = [
    {
        'analysis': 'tonic_pooled_filtered',
        'n_mice': pool_n,
        'W': float(pool_w),
        'p': float(pool_p),
        'dz': float(pool_dz),
        'mean_diff': float(pool_diff.mean()),
        'sd_diff': float(pool_diff.std(ddof=1)),
        'median_n_eng': np.nan,
        'median_n_biased': np.nan,
        'n_mice_predicted_dir': int((pool_diff < 0).sum()),
        'n_mice_opposite_dir': int((pool_diff > 0).sum()),
    }
]
summary_rows.extend(epoch_rows)
summary_rows.extend([
    {
        'analysis': 'tonic_pooled_leave_one_out_min_dz',
        'n_mice': pool_n - 1,
        'W': np.nan,
        'p': np.nan,
        'dz': float(loo_df['dz_without'].min()),
        'mean_diff': np.nan,
        'sd_diff': np.nan,
        'median_n_eng': np.nan,
        'median_n_biased': np.nan,
        'n_mice_predicted_dir': np.nan,
        'n_mice_opposite_dir': np.nan,
    },
    {
        'analysis': 'tonic_pooled_leave_one_out_max_dz',
        'n_mice': pool_n - 1,
        'W': np.nan,
        'p': np.nan,
        'dz': float(loo_df['dz_without'].max()),
        'mean_diff': np.nan,
        'sd_diff': np.nan,
        'median_n_eng': np.nan,
        'median_n_biased': np.nan,
        'n_mice_predicted_dir': np.nan,
        'n_mice_opposite_dir': np.nan,
    },
])
summary_df = pd.DataFrame(summary_rows)

# save
summary_df.to_csv('output/robustness_table.csv', index=False)
loo_df.to_csv('output/robustness_leave_one_out.csv', index=False)
cor_df.to_csv('output/robustness_epoch_correlations.csv', index=False)

print(summary_df.to_string(index=False))
print('\nSaved:')
print('output/robustness_table.csv')
print('output/robustness_leave_one_out.csv')
print('output/robustness_epoch_correlations.csv')



# ===== CELL 45 =====
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
import pandas as pd

def grubbs_test(data, alpha=0.05):
    """Returns boolean mask of outliers via iterative Grubbs' test (two-sided)."""
    x = np.array(data, dtype=float)
    idx = np.arange(len(x))
    mask = np.zeros(len(x), dtype=bool)
    remaining = idx.copy()
    while len(remaining) > 2:
        sub = x[remaining]
        n = len(sub)
        mean, sd = sub.mean(), sub.std(ddof=1)
        abs_dev = np.abs(sub - mean)
        max_idx_local = np.argmax(abs_dev)
        G = abs_dev[max_idx_local] / sd
        t_crit = stats.t.ppf(1 - alpha/(2*n), n-2)
        G_crit = ((n-1)/np.sqrt(n)) * np.sqrt(t_crit**2/(n-2+t_crit**2))
        if G > G_crit:
            mask[remaining[max_idx_local]] = True
            remaining = np.delete(remaining, max_idx_local)
        else:
            break
    return mask

def iqr_test(data, k=1.5):
    x = np.array(data, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return (x < lo) | (x > hi)

def dual_outlier_check(diff_values, subject_ids, alpha=0.05, k=1.5):
    grubbs_mask = grubbs_test(diff_values, alpha=alpha)
    iqr_mask = iqr_test(diff_values, k=k)
    both_mask = grubbs_mask & iqr_mask
    result = []
    for i, sid in enumerate(subject_ids):
        result.append(dict(subject=sid, diff=diff_values[i],
                           grubbs_outlier=grubbs_mask[i], iqr_outlier=iqr_mask[i],
                           dual_outlier=both_mask[i]))
    return result, both_mask



# ===== CELL 46 =====
med = np.median(pool_diff)
mad = np.median(np.abs(pool_diff - med))
thresh = 3 * 1.4826 * mad
outliers = np.abs(pool_diff - med) > thresh
print(f'{outliers.sum()} outliers by 3xMAD rule')
print('pooled result excluding these:', wilcoxon(pool_diff[~outliers]))

subject_ids = valid.index[valid].tolist()
result, both_mask = dual_outlier_check(pool_diff, subject_ids)

result_df = pd.DataFrame(result)
print(result_df.sort_values('diff').to_string(index=False))
print(f'\n{both_mask.sum()} animal(s) flagged as outliers on BOTH Grubbs and IQR')

clean_diff = pool_diff[~both_mask]
stat_clean, p_clean = wilcoxon(clean_diff)
dz_clean = clean_diff.mean() / clean_diff.std(ddof=1)
print(f'\nOriginal (n={len(pool_diff)}): p={pool_p:.4f}, dz={pool_dz:.3f}')
print(f'Dual-outlier-excluded (n={len(clean_diff)}): p={p_clean:.4f}, dz={dz_clean:.3f}')



# ===== CELL 47 =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------
# Terminology
#
# Task context:
#   unbiased   -> 50:50 stimulus-prior block
#   transition -> early trials after entering a biased-prior block
#   stable     -> later trials within a biased-prior block
#
# Latent decision strategy:
#   engaged
#   biased-left / biased-right, pooled as "biased states"
# ---------------------------------------------------------------------

epochs = [
    "unbiased",
    "transition",
    "stable",
]

epoch_display = {
    "unbiased": "Unbiased block\n(50:50 prior)",
    "transition": "Early biased block",
    "stable": "Later biased block",
}

BAD_PUPIL_SUBJECTS = {
    "SH015",
}

colors = {
    "engaged": "tab:blue",
    "biased": "tab:red",
}

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------
# Validate and clean the z-scored source tables
# ---------------------------------------------------------------------

required_epoch_columns = {
    "subject",
    "epoch",
    "eng_mean",
    "biased_mean",
}

missing_epoch_columns = (
    required_epoch_columns
    - set(tp_epoch_z.columns)
)

if missing_epoch_columns:
    raise KeyError(
        "tp_epoch_z is missing required columns: "
        f"{sorted(missing_epoch_columns)}"
    )


tp_epoch_z_plot = (
    tp_epoch_z.loc[
        ~tp_epoch_z["subject"]
        .astype(str)
        .isin(BAD_PUPIL_SUBJECTS)
    ]
    .copy()
)


# Align pooled engaged and biased values by subject.
pooled_z = pd.concat(
    [
        eng_z.rename("engaged"),
        biased_z.rename("biased"),
    ],
    axis=1,
)

pooled_z.index = (
    pooled_z.index.astype(str)
)

pooled_z = pooled_z.loc[
    ~pooled_z.index.isin(
        BAD_PUPIL_SUBJECTS
    )
].dropna(
    subset=[
        "engaged",
        "biased",
    ]
)


# ---------------------------------------------------------------------
# Set up figure
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 6.8)
)

positions = []
data = []
tick_labels = []

# Each tuple:
# engaged position, biased position, p, dz, n, group type
group_pairs = []

# Labels shown beneath each pair.
group_centers = []

pos = 1


# ---------------------------------------------------------------------
# Three task-block phase groups
# ---------------------------------------------------------------------

for epoch in epochs:
    epoch_data = (
        tp_epoch_z_plot.loc[
            tp_epoch_z_plot["epoch"]
            == epoch,
            [
                "subject",
                "eng_mean",
                "biased_mean",
            ],
        ]
        .dropna(
            subset=[
                "eng_mean",
                "biased_mean",
            ]
        )
        .copy()
    )

    # Enforce exactly one paired row per animal and epoch.
    epoch_data = (
        epoch_data
        .groupby(
            "subject",
            as_index=False,
        )
        .agg(
            eng_mean=(
                "eng_mean",
                "mean",
            ),
            biased_mean=(
                "biased_mean",
                "mean",
            ),
        )
    )

    engaged_values = (
        epoch_data["eng_mean"]
        .to_numpy(dtype=float)
    )

    biased_values = (
        epoch_data["biased_mean"]
        .to_numpy(dtype=float)
    )

    if len(engaged_values) == 0:
        print(
            f"Skipping {epoch}: "
            "no complete paired animals."
        )
        continue

    pos_engaged = pos
    pos_biased = pos + 0.8

    data.extend(
        [
            engaged_values,
            biased_values,
        ]
    )

    positions.extend(
        [
            pos_engaged,
            pos_biased,
        ]
    )

    # These labels describe latent GLM-HMM states.
    tick_labels.extend(
        [
            "Engaged\nstate",
            "Biased states\n(L/R pooled)",
        ]
    )

    # This label describes task-block context.
    group_centers.append(
        (
            (
                pos_engaged
                + pos_biased
            ) / 2,
            epoch_display[epoch],
        )
    )

    difference = (
        engaged_values
        - biased_values
    )

    if len(difference) >= 3:
        statistic, p_value = wilcoxon(
            difference
        )

        difference_sd = difference.std(
            ddof=1
        )

        dz = (
            difference.mean()
            / difference_sd
            if difference_sd > 0
            else np.nan
        )

        group_pairs.append(
            (
                pos_engaged,
                pos_biased,
                p_value,
                dz,
                len(difference),
                "epoch",
            )
        )

    pos += 2.2


# ---------------------------------------------------------------------
# Fourth group: pooled across all task blocks
# ---------------------------------------------------------------------

pos += 1.0

engaged_pooled = (
    pooled_z["engaged"]
    .to_numpy(dtype=float)
)

biased_pooled = (
    pooled_z["biased"]
    .to_numpy(dtype=float)
)

if len(engaged_pooled) == 0:
    raise ValueError(
        "No complete pooled z-scored pairs remain."
    )

pos_engaged_pooled = pos
pos_biased_pooled = pos + 0.8

data.extend(
    [
        engaged_pooled,
        biased_pooled,
    ]
)

positions.extend(
    [
        pos_engaged_pooled,
        pos_biased_pooled,
    ]
)

tick_labels.extend(
    [
        "Engaged\nstate",
        "Biased states\n(L/R pooled)",
    ]
)

group_centers.append(
    (
        (
            pos_engaged_pooled
            + pos_biased_pooled
        ) / 2,
        "All task blocks",
    )
)

pooled_difference = (
    engaged_pooled
    - biased_pooled
)

pool_statistic_z, pool_p_z = wilcoxon(
    pooled_difference
)

pool_sd_z = pooled_difference.std(
    ddof=1
)

pool_dz_z = (
    pooled_difference.mean()
    / pool_sd_z
    if pool_sd_z > 0
    else np.nan
)

group_pairs.append(
    (
        pos_engaged_pooled,
        pos_biased_pooled,
        pool_p_z,
        pool_dz_z,
        len(pooled_difference),
        "pooled",
    )
)


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

all_values = np.concatenate(data)

if not np.isfinite(all_values).all():
    raise ValueError(
        "Nonfinite values remain in the "
        "z-scored plotting data."
    )

largest_absolute_z = np.max(
    np.abs(all_values)
)

print(
    "Largest absolute plotted z-score:",
    f"{largest_absolute_z:.3f}",
)

print(
    "Pooled z-scored comparison:",
    f"W={pool_statistic_z:.1f}, "
    f"p={pool_p_z:.6g}, "
    f"dz={pool_dz_z:.3f}, "
    f"n={len(pooled_difference)}",
)

# A value this large would indicate that the z-scored tables
# were rebuilt from stale/corrupted data.
assert largest_absolute_z < 20, (
    "Implausibly large z-scored pupil value remains: "
    f"{largest_absolute_z}"
)


# ---------------------------------------------------------------------
# Draw violins
# ---------------------------------------------------------------------

parts = ax.violinplot(
    data,
    positions=positions,
    widths=0.7,
    showmeans=True,
    showextrema=False,
)

for index, body in enumerate(
    parts["bodies"]
):
    condition = (
        "engaged"
        if index % 2 == 0
        else "biased"
    )

    body.set_facecolor(
        colors[condition]
    )

    body.set_edgecolor(
        colors[condition]
    )

    body.set_alpha(0.35)


# ---------------------------------------------------------------------
# Draw individual animals
# ---------------------------------------------------------------------

for index, values in enumerate(data):
    jitter = rng.uniform(
        -0.12,
        0.12,
        size=len(values),
    )

    condition = (
        "engaged"
        if index % 2 == 0
        else "biased"
    )

    ax.scatter(
        np.full(
            len(values),
            positions[index],
        )
        + jitter,
        values,
        color=colors[condition],
        alpha=0.7,
        s=18,
        zorder=3,
        edgecolor="black",
        linewidth=0.3,
    )


# ---------------------------------------------------------------------
# Statistical annotations
# ---------------------------------------------------------------------

def significance_label(p_value):
    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "n.s."


y_max = max(
    np.max(values)
    for values in data
)

y_min = min(
    np.min(values)
    for values in data
)

y_range = y_max - y_min

if not np.isfinite(y_range) or y_range == 0:
    y_range = 1.0

bracket_height = (
    y_range * 0.06
)


for (
    pos_engaged,
    pos_biased,
    p_value,
    dz,
    n_animals,
    group_type,
) in group_pairs:

    engaged_index = positions.index(
        pos_engaged
    )

    biased_index = positions.index(
        pos_biased
    )

    local_max = max(
        np.max(data[engaged_index]),
        np.max(data[biased_index]),
    )

    bracket_y = (
        local_max
        + bracket_height
    )

    line_width = (
        1.8
        if group_type == "pooled"
        else 1.2
    )

    ax.plot(
        [
            pos_engaged,
            pos_engaged,
            pos_biased,
            pos_biased,
        ],
        [
            bracket_y,
            bracket_y
            + bracket_height * 0.4,
            bracket_y
            + bracket_height * 0.4,
            bracket_y,
        ],
        color="black",
        linewidth=line_width,
    )

    annotation = (
        f"{significance_label(p_value)}\n"
        f"p={p_value:.3f}, "
        f"dz={dz:.2f}, "
        f"n={n_animals}"
    )

    ax.text(
        (
            pos_engaged
            + pos_biased
        ) / 2,
        bracket_y
        + bracket_height * 0.55,
        annotation,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=8.5,
        fontweight=(
            "bold"
            if group_type == "pooled"
            else "normal"
        ),
    )


# ---------------------------------------------------------------------
# Separate task-phase results from pooled result
# ---------------------------------------------------------------------

# The final epoch pair occupies positions 4 and 5.
divider_x = (
    positions[5]
    + pos_engaged_pooled
) / 2

ax.axvline(
    divider_x,
    color="gray",
    linestyle=":",
    linewidth=1,
    alpha=0.6,
)


# ---------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------

ax.set_ylim(
    y_min - y_range * 0.05,
    y_max + y_range * 0.42,
)

ax.set_xticks(
    positions
)

ax.set_xticklabels(
    tick_labels,
    fontsize=9,
)

# Task-context labels beneath each pair.
for center_x, block_label in group_centers:
    ax.text(
        center_x,
        -0.18,
        block_label,
        transform=ax.get_xaxis_transform(),
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9.5,
        fontweight="bold",
        clip_on=False,
    )

ax.axhline(
    0,
    color="black",
    linewidth=0.8,
    alpha=0.35,
)

ax.set_ylabel(
    "Per-animal mean tonic pupil\n"
    "(within-animal z-score)"
)

ax.set_title(
    "Standardized tonic pupil across task-block phases "
    "and latent decision states\n"
    "Paired Wilcoxon signed-rank tests; points represent mice"
)


# ---------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------

handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=colors["engaged"],
        markeredgecolor="none",
        markersize=8,
        label="Engaged GLM-HMM state",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=colors["biased"],
        markeredgecolor="none",
        markersize=8,
        label=(
            "Biased GLM-HMM states "
            "(left/right pooled)"
        ),
    ),
]

ax.legend(
    handles=handles,
    loc="upper right",
    fontsize=9,
)


# Leave room beneath the x-axis for block-phase labels.
plt.tight_layout(
    rect=[
        0,
        0.10,
        1,
        1,
    ]
)

plt.savefig(
    "output/tonic_pupil_zscored_by_epoch_and_pooled_violin.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()



# ===== CELL 48 =====
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp, expit as sigmoid

# -------------------------------------------------------------------
# pupil weighting
# -------------------------------------------------------------------

def pupil_weight_from_tonic(pupil_tonic, floor=0.25, scale=0.5, robust=True):
    """
    Convert trialwise tonic pupil into positive sample weights.
    Default: within-animal robust z-score, then weight = 1 + scale*z, clipped at floor.
    """
    x = np.asarray(pupil_tonic, float)

    if robust:
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        z = (x - med) / (1.4826 * mad) if mad > 0 else np.zeros_like(x)
    else:
        m = np.nanmean(x)
        sd = np.nanstd(x)
        z = (x - m) / sd if sd > 0 else np.zeros_like(x)

    w = 1.0 + scale * z
    w = np.clip(w, floor, None)
    w[~np.isfinite(w)] = 1.0
    return w


# -------------------------------------------------------------------
# design matrix
# -------------------------------------------------------------------

def rightward_choice(df):
    return (df['choice'].to_numpy() == -1).astype(float)

def build_design_matrix_with_pupil(df):
    """
    Ashwood inputs:
      x1 = z-scored signed contrast
      x2 = bias term (ones)
      x3 = previous choice (-1/+1)
      x4 = previous stimulus side (-1/+1)
    plus:
      pupil_w = tonic-pupil-derived trial weight
    """
    df = df.reset_index(drop=True)

    sc = df['signed_contrast'].to_numpy(float)
    sd = np.nanstd(sc)
    scz = sc / sd if sd > 0 else sc

    y = rightward_choice(df)

    pc = np.zeros(len(df))
    pc[1:] = np.where(y[:-1] == 1, 1.0, -1.0)

    ps = np.zeros(len(df))
    ps[1:] = np.sign(sc[:-1])

    X = np.column_stack([scz, np.ones(len(df)), pc, ps])

    if 'pupil_tonic' in df.columns:
        pupil_w = pupil_weight_from_tonic(df['pupil_tonic'].to_numpy(float))
    else:
        pupil_w = np.ones(len(df), dtype=float)

    return X, y, pupil_w


# -------------------------------------------------------------------
# GLM-HMM
# -------------------------------------------------------------------

class GLMHMMWeighted:
    """
    Bernoulli-GLM-HMM fit by EM.
    Same structure as the original pipeline, but the emission M-step
    uses effective weights = gamma[:, k] * pupil_w
    """
    def __init__(self, n_states=3, n_features=4, seed=0,
                 w_prior_var=2.0, sticky_alpha=2.0):
        self.K = n_states
        self.D = n_features
        self.rng = np.random.default_rng(seed)
        self.w_prior_var = w_prior_var
        self.sticky_alpha = sticky_alpha
        self.W = None
        self.A = None
        self.pi = None
        self.ll_history = []

    def init_params(self):
        K, D = self.K, self.D
        self.A = np.full((K, K), 0.02 / max(K - 1, 1))
        np.fill_diagonal(self.A, 0.98)
        self.pi = np.full(K, 1 / K)
        self.W = 0.2 * self.rng.standard_normal((K, D))
        self.W[0, 0] = 3.0  # seed engaged-like state

    def log_emission(self, X, y):
        p1 = np.clip(sigmoid(X @ self.W.T), 1e-9, 1 - 1e-9)  # T x K
        return y[:, None] * np.log(p1) + (1 - y)[:, None] * np.log(1 - p1)

    def forward_backward(self, log_em):
        T, K = log_em.shape
        logA = np.log(self.A + 1e-16)
        logpi = np.log(self.pi + 1e-16)

        la = np.zeros((T, K))
        la[0] = logpi + log_em[0]
        for t in range(1, T):
            la[t] = log_em[t] + logsumexp(la[t - 1][:, None] + logA, axis=0)

        lb = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(logA + log_em[t + 1][None, :] + lb[t + 1][None, :], axis=1)

        ll = logsumexp(la[-1])

        lg = la + lb
        lg -= logsumexp(lg, axis=1, keepdims=True)
        gamma = np.exp(lg)

        xi = np.zeros((K, K))
        for t in range(T - 1):
            m = la[t][:, None] + logA + log_em[t + 1][None, :] + lb[t + 1][None, :]
            xi += np.exp(m - logsumexp(m))

        return gamma, xi, ll

    def posterior(self, X, y):
        return self.forward_backward(self.log_emission(X, y))[0]

    def loglik(self, X, y):
        return self.forward_backward(self.log_emission(X, y))[2]

    def fit(self, X, y, pupil_w=None, n_iter=150, tol=1e-4, verbose=False):
        if pupil_w is None:
            pupil_w = np.ones(len(y), dtype=float)

        self.init_params()
        prev = -np.inf

        for it in range(n_iter):
            gamma, xi, ll = self.forward_backward(self.log_emission(X, y))
            self.ll_history.append(ll)

            # initial state
            self.pi = gamma[0] / gamma[0].sum()

            # transition matrix with sticky prior
            prior_counts = np.eye(self.K) * (self.sticky_alpha - 1)
            post = xi + prior_counts
            self.A = post / post.sum(axis=1, keepdims=True)

            # weighted logistic M-step
            for k in range(self.K):
                w_eff = gamma[:, k] * pupil_w

                if w_eff.sum() < 1e-6:
                    continue
                if (w_eff * y).sum() < 1e-6:
                    continue
                if (w_eff * (1 - y)).sum() < 1e-6:
                    continue

                lr = LogisticRegression(
                    C=self.w_prior_var,
                    fit_intercept=False,
                    max_iter=200,
                    tol=1e-5
                )
                lr.fit(X, y, sample_weight=w_eff)
                self.W[k] = lr.coef_[0]

            if verbose and it % 25 == 0:
                print(f'EM iter {it:3d} loglik {ll:.2f}')

            if abs(ll - prev) < tol:
                break
            prev = ll

        return self


# -------------------------------------------------------------------
# multi-init wrapper
# -------------------------------------------------------------------

def fit_best_weighted(n_states, n_features, X, y, pupil_w,
                      n_init=20, base_seed=0, verbose=False, **fit_kw):
    best_model, best_ll = None, -np.inf
    for i in range(n_init):
        m = GLMHMMWeighted(n_states, n_features, seed=base_seed + i)
        m.fit(X, y, pupil_w=pupil_w, verbose=False, **fit_kw)
        final_ll = m.ll_history[-1] if m.ll_history else -np.inf
        if verbose:
            print(f'init {i:2d}/{n_init} final train loglik {final_ll:.2f}')
        if final_ll > best_ll:
            best_model, best_ll = m, final_ll
    if verbose:
        print(f'-> kept init with train loglik {best_ll:.2f}')
    return best_model


# -------------------------------------------------------------------
# state relabeling
# -------------------------------------------------------------------

STATE_LABELS = ['engaged', 'biased-left', 'biased-right']

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


# -------------------------------------------------------------------
# fit one animal with pupil-weighted GLM-HMM
# -------------------------------------------------------------------

def fit_animal_glmhmm_weighted(trials_list, n_states=3, seed=0, verbose=False, test_frac=0.25, n_init=20):
    """
    trials_list: list of per-session epoch-labelled DataFrames
    Each df must already include:
      signed_contrast, choice, pupil_tonic, epoch
    """
    keep_all, X_all, y_all, pw_all = [], [], [], []

    for df in trials_list:
        d = df[df['choice'] != 0].reset_index(drop=True)
        X, y, pupil_w = build_design_matrix_with_pupil(d)
        keep_all.append(d)
        X_all.append(X)
        y_all.append(y)
        pw_all.append(pupil_w)

    n_sess = len(keep_all)
    rng = np.random.default_rng(seed)

    if n_sess >= 2 and test_frac > 0:
        n_test = max(1, int(round(test_frac * n_sess)))
        test_idx = set(rng.choice(n_sess, size=n_test, replace=False).tolist())
    else:
        test_idx = set()

    train_idx = [i for i in range(n_sess) if i not in test_idx]

    X_tr = np.vstack([X_all[i] for i in train_idx])
    y_tr = np.concatenate([y_all[i] for i in train_idx])
    pw_tr = np.concatenate([pw_all[i] for i in train_idx])

    model = fit_best_weighted(
        n_states=n_states,
        n_features=X_tr.shape[1],
        X=X_tr,
        y=y_tr,
        pupil_w=pw_tr,
        n_init=n_init,
        base_seed=seed,
        verbose=verbose
    )

    order, labels = relabel_states_by_engagement(model)

    def per_trial_ll(idxs):
        if not idxs:
            return np.nan, 0
        ll = sum(model.loglik(X_all[i], y_all[i]) for i in idxs)
        n = sum(len(y_all[i]) for i in idxs)
        return ll / n if n else np.nan, n

    tr_ll, n_tr = per_trial_ll(train_idx)
    te_ll, n_te = per_trial_ll(sorted(test_idx))

    model.cv_ = dict(
        train_ll=tr_ll,
        test_ll=te_ll,
        n_train=n_tr,
        n_test=n_te,
        n_sess=n_sess,
        n_test_sess=len(test_idx)
    )

    out = []
    for d in keep_all:
        X_i, y_i, _ = build_design_matrix_with_pupil(d)
        g = model.posterior(X_i, y_i)[:, order]

        d = d.copy()
        for k in range(n_states):
            d[f'p_state{k}'] = g[:, k]
        d['state'] = np.argmax(g, axis=1)
        d['state_label'] = [labels[s] for s in d['state']]
        out.append(d)

    return model, pd.concat(out, ignore_index=True), order


# -------------------------------------------------------------------
# example usage
# -------------------------------------------------------------------
# trials_list must already include pupil_tonic per trial
#
# demo_trials = [label_trial_epochs(load_session_trials(e)) for e in subj_sessions[demo_subj]]
# then merge/add pupil_tonic per session exactly as in your existing pipeline before calling:
#
# model_w, demo_df_w, order_w = fit_animal_glmhmm_weighted(
#     demo_trials,
#     n_states=3,
#     seed=0,
#     verbose=True,
#     test_frac=0.25,
#     n_init=20
# )
#
# print('weighted model CV:', model_w.cv_)
# print('weights ordered engaged, biased-left, biased-right:')
# print(np.round(model_w.W[order_w], 2))



# ===== CELL 49 =====
def fit_best_weighted_fast(n_states, n_features, X, y, pupil_w,
                           n_init=20, n_keep=3,
                           screen_iter=25, full_iter=150,
                           base_seed=0, verbose=False):
    screened = []

    # stage 1: cheap screening
    for i in range(n_init):
        m = GLMHMMWeighted(n_states, n_features, seed=base_seed + i)
        m.fit(X, y, pupil_w=pupil_w, n_iter=screen_iter, tol=1e-3, verbose=False)
        ll = m.ll_history[-1] if m.ll_history else -np.inf
        screened.append((ll, base_seed + i))
        if verbose:
            print(f'screen init {i:2d}/{n_init} ll={ll:.2f}')

    screened.sort(reverse=True, key=lambda z: z[0])
    keep = screened[:n_keep]

    # stage 2: full fit on best few seeds
    best_model, best_ll = None, -np.inf
    for j, (_, seed) in enumerate(keep):
        m = GLMHMMWeighted(n_states, n_features, seed=seed)
        m.fit(X, y, pupil_w=pupil_w, n_iter=full_iter, tol=1e-4, verbose=False)
        ll = m.ll_history[-1] if m.ll_history else -np.inf
        if verbose:
            print(f'full init {j+1:2d}/{n_keep} seed={seed} ll={ll:.2f}')
        if ll > best_ll:
            best_model, best_ll = m, ll

    if verbose:
        print(f'-> kept seed with train loglik {best_ll:.2f}')
    return best_model



# ===== CELL 50 =====
# =====================================================================
# GLM-HMM POSTERIOR UNCERTAINTY AND STATE-LABILITY METRICS
#
# Creates:
#   posterior_entropy       uncertainty about the current state
#   js_to_next              magnitude of posterior change on next trial
#   hard_switch_to_next     whether the argmax state changes next trial
#
# It also creates three-trial future outcomes aligned to trial t:
#   future_entropy_max_3
#   future_js_max_3
#   future_switch_any_3
#
# No GLM-HMM refitting or pupil extraction is performed.
# =====================================================================

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Start from the existing trial table
# ---------------------------------------------------------------------

if "pupil_trials_all" in globals():
    lability_df = pupil_trials_all.copy()
elif "trials_df" in globals():
    lability_df = trials_df.copy()
else:
    raise NameError(
        "Neither pupil_trials_all nor trials_df exists."
    )


# SH015 has unusable pupil data and should not enter pupil analyses.
lability_df = lability_df.loc[
    lability_df["subject"].astype(str) != "SH015"
].copy()

lability_df = lability_df.reset_index(drop=True)

posterior_columns = [
    "p_state0",
    "p_state1",
    "p_state2",
]

missing_columns = [
    column
    for column in posterior_columns
    if column not in lability_df.columns
]

if missing_columns:
    raise KeyError(
        "Missing posterior columns: "
        f"{missing_columns}"
    )


# ---------------------------------------------------------------------
# 2. Preserve trial order and detect obvious session boundaries
#
# Session timestamps normally reset when a new recording begins.
# This avoids measuring a switch between the final trial of one session
# and the first trial of the next, without needing eid metadata.
# ---------------------------------------------------------------------

lability_df["_row_in_subject"] = (
    lability_df
    .groupby(
        "subject",
        sort=False,
    )
    .cumcount()
)

if "stimOn_times" in lability_df.columns:
    current_stim_time = pd.to_numeric(
        lability_df["stimOn_times"],
        errors="coerce",
    )

    previous_stim_time = (
        current_stim_time
        .groupby(
            lability_df["subject"],
            sort=False,
        )
        .shift(1)
    )

    sequence_break = (
        previous_stim_time.isna()
        | current_stim_time.isna()
        | previous_stim_time.isna()
        | current_stim_time.le(
            previous_stim_time
        )
    )

else:
    # Fallback: only identify the first row of each animal.
    sequence_break = (
        lability_df["_row_in_subject"]
        == 0
    )

lability_df["_sequence_break"] = (
    sequence_break
)

lability_df["sequence_id"] = (
    lability_df
    .groupby(
        "subject",
        sort=False,
    )["_sequence_break"]
    .cumsum()
    .astype(int)
    - 1
)

sequence_group_columns = [
    "subject",
    "sequence_id",
]


# ---------------------------------------------------------------------
# 3. Sanitize and normalize posterior probabilities
# ---------------------------------------------------------------------

posterior = (
    lability_df[posterior_columns]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(dtype=float)
)

valid_posterior = (
    np.isfinite(posterior).all(axis=1)
    & (posterior.sum(axis=1) > 0)
)

normalized_posterior = np.full_like(
    posterior,
    np.nan,
    dtype=float,
)

normalized_posterior[
    valid_posterior
] = (
    posterior[valid_posterior]
    / posterior[
        valid_posterior
    ].sum(
        axis=1,
        keepdims=True,
    )
)

# Protect log calculations without materially changing probabilities.
epsilon = 1e-12

clipped_posterior = np.clip(
    normalized_posterior,
    epsilon,
    1.0,
)


# ---------------------------------------------------------------------
# 4. Current-trial posterior entropy
#
# Dividing by log(K) normalizes entropy to [0, 1]:
#   0 = one state has essentially all posterior probability
#   1 = equal uncertainty among all three states
# ---------------------------------------------------------------------

n_states = len(
    posterior_columns
)

posterior_entropy = (
    -np.sum(
        clipped_posterior
        * np.log(clipped_posterior),
        axis=1,
    )
    / np.log(n_states)
)

posterior_entropy[
    ~valid_posterior
] = np.nan

lability_df["posterior_entropy"] = (
    posterior_entropy
)

lability_df["posterior_confidence"] = (
    np.nanmax(
        normalized_posterior,
        axis=1,
    )
)


# ---------------------------------------------------------------------
# 5. Hard state assignment
# ---------------------------------------------------------------------

hard_state = np.full(
    len(lability_df),
    -1,
    dtype=int,
)

hard_state[
    valid_posterior
] = np.argmax(
    normalized_posterior[
        valid_posterior
    ],
    axis=1,
)

state_names = np.array(
    [
        "engaged",
        "biased-left",
        "biased-right",
    ],
    dtype=object,
)

hard_state_label = np.full(
    len(lability_df),
    None,
    dtype=object,
)

hard_state_label[
    valid_posterior
] = state_names[
    hard_state[valid_posterior]
]

lability_df["hard_state"] = pd.array(
    np.where(
        valid_posterior,
        hard_state,
        np.nan,
    ),
    dtype="Int64",
)

lability_df["hard_state_label"] = (
    hard_state_label
)


# Verify consistency with the notebook's existing state assignment.
if "state" in lability_df.columns:
    existing_state = pd.to_numeric(
        lability_df["state"],
        errors="coerce",
    ).to_numpy(dtype=float)

    comparable = (
        valid_posterior
        & np.isfinite(existing_state)
    )

    state_match_fraction = np.mean(
        hard_state[comparable]
        == existing_state[
            comparable
        ].astype(int)
    )

    print(
        "Recomputed argmax agrees with existing state on "
        f"{state_match_fraction:.3%} of comparable trials."
    )


# ---------------------------------------------------------------------
# 6. Next-trial posterior, respecting detected sequence boundaries
# ---------------------------------------------------------------------

for state_index, column in enumerate(
    posterior_columns
):
    lability_df[
        f"_next_{column}"
    ] = (
        lability_df
        .groupby(
            sequence_group_columns,
            sort=False,
        )[column]
        .shift(-1)
    )

next_posterior = (
    lability_df[
        [
            f"_next_{column}"
            for column in posterior_columns
        ]
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(dtype=float)
)

valid_next_posterior = (
    np.isfinite(next_posterior).all(axis=1)
    & (next_posterior.sum(axis=1) > 0)
)

normalized_next_posterior = np.full_like(
    next_posterior,
    np.nan,
    dtype=float,
)

normalized_next_posterior[
    valid_next_posterior
] = (
    next_posterior[
        valid_next_posterior
    ]
    / next_posterior[
        valid_next_posterior
    ].sum(
        axis=1,
        keepdims=True,
    )
)

valid_posterior_pair = (
    valid_posterior
    & valid_next_posterior
)


# ---------------------------------------------------------------------
# 7. Jensen-Shannon divergence from trial t to trial t+1
#
# This captures posterior movement even when both trials have low entropy.
#
# Normalized by log(2), giving a range of approximately [0, 1].
# ---------------------------------------------------------------------

p_t = np.clip(
    normalized_posterior,
    epsilon,
    1.0,
)

p_next = np.clip(
    normalized_next_posterior,
    epsilon,
    1.0,
)

midpoint = (
    0.5
    * (
        p_t
        + p_next
    )
)

js_divergence = np.full(
    len(lability_df),
    np.nan,
    dtype=float,
)

js_divergence[
    valid_posterior_pair
] = (
    0.5
    * np.sum(
        p_t[valid_posterior_pair]
        * np.log(
            p_t[valid_posterior_pair]
            / midpoint[
                valid_posterior_pair
            ]
        ),
        axis=1,
    )
    + 0.5
    * np.sum(
        p_next[
            valid_posterior_pair
        ]
        * np.log(
            p_next[
                valid_posterior_pair
            ]
            / midpoint[
                valid_posterior_pair
            ]
        ),
        axis=1,
    )
) / np.log(2)

lability_df["js_to_next"] = (
    js_divergence
)


# ---------------------------------------------------------------------
# 8. Entropy change and hard switch from t to t+1
# ---------------------------------------------------------------------

lability_df["entropy_next"] = (
    lability_df
    .groupby(
        sequence_group_columns,
        sort=False,
    )["posterior_entropy"]
    .shift(-1)
)

lability_df["entropy_change_to_next"] = (
    lability_df["entropy_next"]
    - lability_df["posterior_entropy"]
)

lability_df["hard_state_next"] = (
    lability_df
    .groupby(
        sequence_group_columns,
        sort=False,
    )["hard_state"]
    .shift(-1)
)

hard_switch = np.full(
    len(lability_df),
    np.nan,
    dtype=float,
)

valid_hard_state_pair = (
    lability_df["hard_state"].notna()
    & lability_df["hard_state_next"].notna()
)

hard_switch[
    valid_hard_state_pair
] = (
    lability_df.loc[
        valid_hard_state_pair,
        "hard_state",
    ].astype(int).to_numpy()
    != lability_df.loc[
        valid_hard_state_pair,
        "hard_state_next",
    ].astype(int).to_numpy()
).astype(float)

lability_df["hard_switch_to_next"] = (
    hard_switch
)


# Direction-specific transitions, useful later.
current_state = (
    lability_df["hard_state"]
)

next_state = (
    lability_df["hard_state_next"]
)

lability_df["switch_to_engaged"] = np.where(
    valid_hard_state_pair,
    (
        current_state.ne(0)
        & next_state.eq(0)
    ).astype(float),
    np.nan,
)

lability_df["switch_from_engaged"] = np.where(
    valid_hard_state_pair,
    (
        current_state.eq(0)
        & next_state.ne(0)
    ).astype(float),
    np.nan,
)


# ---------------------------------------------------------------------
# 9. Create prospective three-trial lability outcomes
#
# These are aligned with predictor values on trial t.
# ---------------------------------------------------------------------

future_entropy_columns = []
future_js_columns = []
future_switch_columns = []

for lag in range(1, 4):
    entropy_column = (
        f"_entropy_plus_{lag}"
    )

    lability_df[entropy_column] = (
        lability_df
        .groupby(
            sequence_group_columns,
            sort=False,
        )["posterior_entropy"]
        .shift(-lag)
    )

    future_entropy_columns.append(
        entropy_column
    )


# JS at t already describes t -> t+1.
# Shifting by -1 gives t+1 -> t+2, etc.
for transition_offset in range(3):
    js_column = (
        f"_js_transition_plus_"
        f"{transition_offset}"
    )

    switch_column = (
        f"_switch_transition_plus_"
        f"{transition_offset}"
    )

    lability_df[js_column] = (
        lability_df
        .groupby(
            sequence_group_columns,
            sort=False,
        )["js_to_next"]
        .shift(-transition_offset)
    )

    lability_df[switch_column] = (
        lability_df
        .groupby(
            sequence_group_columns,
            sort=False,
        )["hard_switch_to_next"]
        .shift(-transition_offset)
    )

    future_js_columns.append(
        js_column
    )

    future_switch_columns.append(
        switch_column
    )


lability_df["future_entropy_max_3"] = (
    lability_df[
        future_entropy_columns
    ]
    .max(
        axis=1,
        skipna=True,
    )
)

lability_df["future_entropy_mean_3"] = (
    lability_df[
        future_entropy_columns
    ]
    .mean(
        axis=1,
        skipna=True,
    )
)

lability_df["future_js_max_3"] = (
    lability_df[
        future_js_columns
    ]
    .max(
        axis=1,
        skipna=True,
    )
)

# For 0/1 values, maximum means "any switch."
lability_df["future_switch_any_3"] = (
    lability_df[
        future_switch_columns
    ]
    .max(
        axis=1,
        skipna=True,
    )
)


# ---------------------------------------------------------------------
# 10. Attach clean pupil predictors
# ---------------------------------------------------------------------

if (
    "pupil_phasic" in lability_df.columns
    and "pupil_phasic_ok"
    in lability_df.columns
):
    lability_df["pupil_phasic_clean"] = (
        lability_df["pupil_phasic"]
        .where(
            lability_df[
                "pupil_phasic_ok"
            ]
        )
    )

elif "pupil_phasic" in lability_df.columns:
    lability_df["pupil_phasic_clean"] = (
        lability_df["pupil_phasic"]
    )


if (
    "pupil_tonic" in lability_df.columns
    and "pupil_tonic_ok"
    in lability_df.columns
):
    lability_df["pupil_tonic_clean"] = (
        lability_df["pupil_tonic"]
        .where(
            lability_df[
                "pupil_tonic_ok"
            ]
        )
    )

elif "pupil_tonic" in lability_df.columns:
    lability_df["pupil_tonic_clean"] = (
        lability_df["pupil_tonic"]
    )


# ---------------------------------------------------------------------
# 11. Sanity checks and descriptive summary
# ---------------------------------------------------------------------

assert (
    lability_df["posterior_entropy"]
    .dropna()
    .between(0, 1)
    .all()
), "Posterior entropy fell outside [0, 1]."

assert (
    lability_df["js_to_next"]
    .dropna()
    .between(0, 1)
    .all()
), "Jensen-Shannon divergence fell outside [0, 1]."

print(
    "\nTrials:",
    f"{len(lability_df):,}",
)

print(
    "Detected animal/session sequences:",
    lability_df[
        [
            "subject",
            "sequence_id",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)

print(
    "\nPosterior entropy:"
)

print(
    lability_df[
        "posterior_entropy"
    ]
    .describe()
    .round(4)
)

print(
    "\nJensen-Shannon change to next trial:"
)

print(
    lability_df[
        "js_to_next"
    ]
    .describe()
    .round(4)
)

print(
    "\nHard-switch rate to next trial:",
    f"{lability_df['hard_switch_to_next'].mean():.3%}",
)

print(
    "Any hard switch in next three transitions:",
    f"{lability_df['future_switch_any_3'].mean():.3%}",
)

if "pupil_phasic_clean" in lability_df.columns:
    print(
        "Trials with clean phasic pupil:",
        f"{lability_df['pupil_phasic_clean'].notna().sum():,}",
    )


# Remove temporary next-posterior columns.
lability_df = lability_df.drop(
    columns=[
        column
        for column in lability_df.columns
        if column.startswith("_next_p_state")
    ],
    errors="ignore",
)



# ===== CELL 51 =====
# =====================================================================
# REPAIR POSTERIOR NORMALIZATION AND JENSEN-SHANNON DIVERGENCE
#
# Run after the failed lability cell. It overwrites the affected columns
# without refitting the GLM-HMM.
# =====================================================================

import numpy as np
import pandas as pd
from scipy.special import rel_entr


posterior_columns = [
    "p_state0",
    "p_state1",
    "p_state2",
]

sequence_group_columns = [
    "subject",
    "sequence_id",
]


# ---------------------------------------------------------------------
# 1. Probability sanitization helper
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# 2. Inspect the original posterior values
# ---------------------------------------------------------------------

raw_current = (
    lability_df[posterior_columns]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(dtype=float)
)

finite_raw = raw_current[
    np.isfinite(raw_current)
]

print("Raw posterior diagnostics:")

if finite_raw.size:
    print(
        "  minimum entry:",
        f"{finite_raw.min():.16g}",
    )

    print(
        "  maximum entry:",
        f"{finite_raw.max():.16g}",
    )

print(
    "  negative entries:",
    int(
        np.sum(
            np.isfinite(raw_current)
            & (raw_current < 0)
        )
    ),
)

raw_row_sums = np.nansum(
    raw_current,
    axis=1,
)

finite_sum_mask = np.isfinite(
    raw_row_sums
)

if finite_sum_mask.any():
    print(
        "  raw row-sum range:",
        f"{raw_row_sums[finite_sum_mask].min():.16g}",
        "to",
        f"{raw_row_sums[finite_sum_mask].max():.16g}",
    )


# ---------------------------------------------------------------------
# 3. Sanitize current-trial posterior probabilities
# ---------------------------------------------------------------------

posterior, valid_posterior = (
    sanitize_probability_matrix(
        raw_current
    )
)

n_states = posterior.shape[1]


# ---------------------------------------------------------------------
# 4. Recompute normalized posterior entropy
# ---------------------------------------------------------------------

posterior_entropy = np.full(
    len(lability_df),
    np.nan,
    dtype=float,
)

# rel_entr(p, 1) equals p * log(p), with the correct 0 log 0 limit.
posterior_entropy[
    valid_posterior
] = (
    -np.sum(
        rel_entr(
            posterior[valid_posterior],
            np.ones_like(
                posterior[valid_posterior]
            ),
        ),
        axis=1,
    )
    / np.log(n_states)
)

# Clean tiny numerical excursions at the boundaries.
posterior_entropy[
    valid_posterior
] = np.clip(
    posterior_entropy[
        valid_posterior
    ],
    0.0,
    1.0,
)

lability_df["posterior_entropy"] = (
    posterior_entropy
)

posterior_confidence = np.full(
    len(lability_df),
    np.nan,
    dtype=float,
)

posterior_confidence[
    valid_posterior
] = np.max(
    posterior[valid_posterior],
    axis=1,
)

lability_df["posterior_confidence"] = (
    posterior_confidence
)


# ---------------------------------------------------------------------
# 5. Obtain and sanitize next-trial probabilities
# ---------------------------------------------------------------------

next_raw = np.column_stack(
    [
        lability_df
        .groupby(
            sequence_group_columns,
            sort=False,
        )[column]
        .shift(-1)
        .to_numpy(dtype=float)

        for column in posterior_columns
    ]
)

next_posterior, valid_next_posterior = (
    sanitize_probability_matrix(
        next_raw
    )
)

valid_posterior_pair = (
    valid_posterior
    & valid_next_posterior
)


# ---------------------------------------------------------------------
# 6. Recompute Jensen-Shannon divergence
#
# JSD = 0.5 KL(P || M) + 0.5 KL(Q || M), M = (P + Q) / 2
#
# Dividing by log(2) normalizes the divergence to [0, 1].
# scipy.special.rel_entr correctly handles probabilities equal to zero.
# ---------------------------------------------------------------------

js_to_next = np.full(
    len(lability_df),
    np.nan,
    dtype=float,
)

p = posterior[
    valid_posterior_pair
]

q = next_posterior[
    valid_posterior_pair
]

midpoint = (
    p + q
) / 2.0

js_raw = (
    0.5
    * np.sum(
        rel_entr(
            p,
            midpoint,
        ),
        axis=1,
    )
    + 0.5
    * np.sum(
        rel_entr(
            q,
            midpoint,
        ),
        axis=1,
    )
) / np.log(2.0)

print("\nRecomputed raw JSD range:")

if js_raw.size:
    print(
        f"  minimum: {js_raw.min():.16g}"
    )

    print(
        f"  maximum: {js_raw.max():.16g}"
    )

# Only numerical excursions should remain after proper normalization.
large_violation = (
    (js_raw < -1e-10)
    | (js_raw > 1 + 1e-10)
)

if large_violation.any():
    bad_pair_indices = np.flatnonzero(
        valid_posterior_pair
    )[large_violation]

    diagnostic_columns = [
        "subject",
        "sequence_id",
    ]

    if "stimOn_times" in lability_df.columns:
        diagnostic_columns.append(
            "stimOn_times"
        )

    diagnostic = (
        lability_df.loc[
            bad_pair_indices,
            diagnostic_columns,
        ]
        .copy()
    )

    diagnostic["js_raw"] = (
        js_raw[large_violation]
    )

    print(
        "\nUnexpected JSD violations:"
    )

    print(
        diagnostic
        .head(20)
        .to_string(index=False)
    )

    raise ValueError(
        "Jensen-Shannon divergence remains materially "
        "outside [0, 1] after probability normalization."
    )

# Clip only floating-point noise, such as 1.0000000000000002.
js_raw = np.clip(
    js_raw,
    0.0,
    1.0,
)

js_to_next[
    valid_posterior_pair
] = js_raw

lability_df["js_to_next"] = (
    js_to_next
)


# ---------------------------------------------------------------------
# 7. Refresh entropy-change metrics
# ---------------------------------------------------------------------

lability_df["entropy_next"] = (
    lability_df
    .groupby(
        sequence_group_columns,
        sort=False,
    )["posterior_entropy"]
    .shift(-1)
)

lability_df["entropy_change_to_next"] = (
    lability_df["entropy_next"]
    - lability_df["posterior_entropy"]
)


# ---------------------------------------------------------------------
# 8. Refresh the three-trial future metrics
# ---------------------------------------------------------------------

future_entropy_columns = []

for lag in range(1, 4):
    column = f"_entropy_plus_{lag}"

    lability_df[column] = (
        lability_df
        .groupby(
            sequence_group_columns,
            sort=False,
        )["posterior_entropy"]
        .shift(-lag)
    )

    future_entropy_columns.append(
        column
    )

lability_df["future_entropy_max_3"] = (
    lability_df[
        future_entropy_columns
    ]
    .max(
        axis=1,
        skipna=True,
    )
)

lability_df["future_entropy_mean_3"] = (
    lability_df[
        future_entropy_columns
    ]
    .mean(
        axis=1,
        skipna=True,
    )
)


future_js_columns = []

for transition_offset in range(3):
    column = (
        f"_js_transition_plus_"
        f"{transition_offset}"
    )

    lability_df[column] = (
        lability_df
        .groupby(
            sequence_group_columns,
            sort=False,
        )["js_to_next"]
        .shift(-transition_offset)
    )

    future_js_columns.append(
        column
    )

lability_df["future_js_max_3"] = (
    lability_df[
        future_js_columns
    ]
    .max(
        axis=1,
        skipna=True,
    )
)


# ---------------------------------------------------------------------
# 9. Tolerant assertions
# ---------------------------------------------------------------------

entropy_values = (
    lability_df[
        "posterior_entropy"
    ]
    .dropna()
    .to_numpy(dtype=float)
)

js_values = (
    lability_df[
        "js_to_next"
    ]
    .dropna()
    .to_numpy(dtype=float)
)

assert np.all(
    entropy_values >= -1e-12
), "Posterior entropy contains negative values."

assert np.all(
    entropy_values <= 1 + 1e-12
), "Posterior entropy exceeds one."

assert np.all(
    js_values >= -1e-12
), "Jensen-Shannon divergence contains negative values."

assert np.all(
    js_values <= 1 + 1e-12
), "Jensen-Shannon divergence exceeds one."


# ---------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------

print(
    "\nPosterior entropy:"
)

print(
    lability_df[
        "posterior_entropy"
    ]
    .describe()
    .round(5)
)

print(
    "\nJensen-Shannon divergence to next trial:"
)

print(
    lability_df[
        "js_to_next"
    ]
    .describe()
    .round(5)
)

print(
    "\nJSD repair completed successfully."
)



# ===== CELL 52 =====
# =====================================================================
# DOES PHASIC PUPIL PREDICT SUBSEQUENT GLM-HMM STATE LABILITY?
#
# Outcomes:
#   1. future_js_max_3
#        Largest posterior movement over the next 3 transitions
#
#   2. future_switch_any_3
#        Any hard state switch over the next 3 transitions
#
#   3. future_to_engaged_any_3
#        For currently biased trials only:
#        whether the model enters the engaged state in the next 3 trials
#
# Predictors:
#   - within-animal phasic pupil
#   - negative feedback
#   - pupil × negative-feedback interaction
#   - current posterior entropy
#   - current inferred state
#   - stimulus strength
#   - task block
#   - approximate position within the recording sequence
#
# Models use subject-clustered standard errors.
# No GLM-HMM refitting is performed.
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------
# 1. Create the analysis dataframe
# ---------------------------------------------------------------------

analysis_df = lability_df.copy()

analysis_df = analysis_df.loc[
    analysis_df["subject"].astype(str) != "SH015"
].copy()


# ---------------------------------------------------------------------
# 2. Clean phasic pupil predictor
# ---------------------------------------------------------------------

if "pupil_phasic_clean" not in analysis_df.columns:
    if (
        "pupil_phasic_ok" in analysis_df.columns
        and "pupil_phasic" in analysis_df.columns
    ):
        analysis_df["pupil_phasic_clean"] = (
            analysis_df["pupil_phasic"]
            .where(
                analysis_df["pupil_phasic_ok"]
            )
        )

    elif "pupil_phasic" in analysis_df.columns:
        analysis_df["pupil_phasic_clean"] = (
            analysis_df["pupil_phasic"]
        )

    else:
        raise KeyError(
            "No phasic pupil column was found."
        )


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


# Standardize within each animal so camera scale and baseline pupil
# differences do not determine the effect.
analysis_df["pupil_phasic_z"] = (
    analysis_df
    .groupby(
        "subject",
        sort=False,
    )["pupil_phasic_clean"]
    .transform(
        robust_zscore_series
    )
)


# ---------------------------------------------------------------------
# 3. Define negative feedback
#
# In the IBL trial table:
#   feedbackType == 1  means rewarded/correct
#   other finite values represent unrewarded/incorrect feedback
# ---------------------------------------------------------------------

if "feedbackType" not in analysis_df.columns:
    raise KeyError(
        "feedbackType is required for the failure analysis."
    )

feedback = pd.to_numeric(
    analysis_df["feedbackType"],
    errors="coerce",
)

analysis_df["negative_feedback"] = np.where(
    feedback.notna(),
    feedback.ne(1).astype(float),
    np.nan,
)


# ---------------------------------------------------------------------
# 4. Approximate stimulus strength
#
# Only one of contrastLeft / contrastRight is ordinarily populated.
# ---------------------------------------------------------------------

if {
    "contrastLeft",
    "contrastRight",
}.issubset(analysis_df.columns):

    contrast_left = pd.to_numeric(
        analysis_df["contrastLeft"],
        errors="coerce",
    ).abs()

    contrast_right = pd.to_numeric(
        analysis_df["contrastRight"],
        errors="coerce",
    ).abs()

    analysis_df["stimulus_strength"] = np.maximum(
        contrast_left.fillna(0),
        contrast_right.fillna(0),
    )

else:
    analysis_df["stimulus_strength"] = 0.0


# ---------------------------------------------------------------------
# 5. Position within each detected recording sequence
#
# This is included only as a nuisance covariate. It is not treated as
# the primary explanation.
# ---------------------------------------------------------------------

sequence_groups = [
    "subject",
    "sequence_id",
]

analysis_df["trial_in_sequence"] = (
    analysis_df
    .groupby(
        sequence_groups,
        sort=False,
    )
    .cumcount()
)

sequence_size = (
    analysis_df
    .groupby(
        sequence_groups,
        sort=False,
    )["trial_in_sequence"]
    .transform("size")
)

analysis_df["trial_fraction"] = np.where(
    sequence_size > 1,
    analysis_df["trial_in_sequence"]
    / (sequence_size - 1),
    0.0,
)


# ---------------------------------------------------------------------
# 6. Create a direction-specific outcome:
#    biased state at t -> engaged state within the next 3 trials
# ---------------------------------------------------------------------

future_engaged_columns = []

for lag in range(1, 4):
    future_state = (
        analysis_df
        .groupby(
            sequence_groups,
            sort=False,
        )["hard_state"]
        .shift(-lag)
    )

    column = (
        f"_engaged_at_plus_{lag}"
    )

    analysis_df[column] = np.where(
        future_state.notna(),
        future_state.eq(0).astype(float),
        np.nan,
    )

    future_engaged_columns.append(
        column
    )


analysis_df["future_to_engaged_any_3"] = (
    analysis_df[
        future_engaged_columns
    ]
    .max(
        axis=1,
        skipna=True,
    )
)


# ---------------------------------------------------------------------
# 7. Transform JSD for regression
#
# Raw JSD is strongly right-skewed. The monotonic log transform retains
# the ordering while reducing the influence of the rare values near 1.
# ---------------------------------------------------------------------

analysis_df["future_js_log"] = np.log1p(
    100
    * analysis_df["future_js_max_3"]
)


# ---------------------------------------------------------------------
# 8. Restrict to rows with usable predictors
# ---------------------------------------------------------------------

common_required = [
    "subject",
    "pupil_phasic_z",
    "negative_feedback",
    "posterior_entropy",
    "hard_state_label",
    "stimulus_strength",
    "trial_fraction",
]

if "probabilityLeft" not in analysis_df.columns:
    # Create a harmless single-level block variable if unavailable.
    analysis_df["probabilityLeft"] = 0.5


# Remove extremely large z-scores that can arise in animals with nearly
# zero phasic variance. This is a numerical guard, not the main QC step.
analysis_df.loc[
    analysis_df["pupil_phasic_z"].abs() > 10,
    "pupil_phasic_z",
] = np.nan


# ---------------------------------------------------------------------
# 9. Continuous state-lability model
#
# Outcome: largest JSD during the next three transitions
# ---------------------------------------------------------------------

js_model_df = (
    analysis_df
    .dropna(
        subset=common_required
        + [
            "future_js_log",
            "probabilityLeft",
        ]
    )
    .copy()
)

js_formula = """
future_js_log
~ pupil_phasic_z * negative_feedback
+ posterior_entropy
+ C(hard_state_label)
+ stimulus_strength
+ C(probabilityLeft)
+ trial_fraction
"""

js_model = (
    smf.ols(
        formula=js_formula,
        data=js_model_df,
    )
    .fit(
        cov_type="cluster",
        cov_kwds={
            "groups": js_model_df["subject"],
        },
    )
)


# ---------------------------------------------------------------------
# 10. Any state switch during the next three transitions
# ---------------------------------------------------------------------

switch_model_df = (
    analysis_df
    .dropna(
        subset=common_required
        + [
            "future_switch_any_3",
            "probabilityLeft",
        ]
    )
    .copy()
)

switch_model_df[
    "future_switch_any_3"
] = (
    switch_model_df[
        "future_switch_any_3"
    ].astype(int)
)

switch_formula = """
future_switch_any_3
~ pupil_phasic_z * negative_feedback
+ posterior_entropy
+ C(hard_state_label)
+ stimulus_strength
+ C(probabilityLeft)
+ trial_fraction
"""

switch_model = (
    smf.glm(
        formula=switch_formula,
        data=switch_model_df,
        family=sm.families.Binomial(),
    )
    .fit(
        cov_type="cluster",
        cov_kwds={
            "groups": switch_model_df["subject"],
        },
    )
)


# ---------------------------------------------------------------------
# 11. Direction-specific model:
#     currently biased -> engaged within the next three trials
# ---------------------------------------------------------------------

return_model_df = (
    analysis_df.loc[
        analysis_df["hard_state"].isin(
            [
                1,
                2,
            ]
        )
    ]
    .dropna(
        subset=common_required
        + [
            "future_to_engaged_any_3",
            "probabilityLeft",
        ]
    )
    .copy()
)

return_model_df[
    "future_to_engaged_any_3"
] = (
    return_model_df[
        "future_to_engaged_any_3"
    ].astype(int)
)

return_formula = """
future_to_engaged_any_3
~ pupil_phasic_z * negative_feedback
+ posterior_entropy
+ C(hard_state_label)
+ stimulus_strength
+ C(probabilityLeft)
+ trial_fraction
"""

return_model = (
    smf.glm(
        formula=return_formula,
        data=return_model_df,
        family=sm.families.Binomial(),
    )
    .fit(
        cov_type="cluster",
        cov_kwds={
            "groups": return_model_df["subject"],
        },
    )
)


# ---------------------------------------------------------------------
# 12. Print only the coefficients central to the hypothesis
# ---------------------------------------------------------------------

def print_hypothesis_terms(
    model,
    model_name,
    logistic=False,
):
    terms = [
        "pupil_phasic_z",
        "negative_feedback",
        "pupil_phasic_z:negative_feedback",
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(model_name)

    print(
        "=" * 72
    )

    table_rows = []

    for term in terms:
        if term not in model.params.index:
            continue

        coefficient = model.params[term]
        standard_error = model.bse[term]
        p_value = model.pvalues[term]

        row = {
            "term": term,
            "coefficient": coefficient,
            "SE": standard_error,
            "p": p_value,
        }

        if logistic:
            row["odds_ratio"] = np.exp(
                coefficient
            )

        table_rows.append(row)

    print(
        pd.DataFrame(
            table_rows
        ).to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.5f}"
            ),
        )
    )

    # Total pupil slope specifically following negative feedback:
    # main pupil effect + interaction effect.
    if {
        "pupil_phasic_z",
        "pupil_phasic_z:negative_feedback",
    }.issubset(model.params.index):

        failure_slope_test = model.t_test(
            "pupil_phasic_z "
            "+ pupil_phasic_z:negative_feedback = 0"
        )

        estimated_failure_slope = (
            model.params["pupil_phasic_z"]
            + model.params[
                "pupil_phasic_z:negative_feedback"
            ]
        )

        print(
            "\nTotal pupil slope on negative-feedback trials:"
        )

        print(
            f"  coefficient = "
            f"{estimated_failure_slope:.5f}"
        )

        print(
            f"  p = "
            f"{float(failure_slope_test.pvalue):.6g}"
        )

        if logistic:
            print(
                f"  odds ratio per 1-SD pupil increase = "
                f"{np.exp(estimated_failure_slope):.4f}"
            )


print_hypothesis_terms(
    js_model,
    (
        "Future posterior lability "
        "(log-transformed maximum JSD)"
    ),
    logistic=False,
)

print_hypothesis_terms(
    switch_model,
    (
        "Any GLM-HMM state switch "
        "within the next 3 transitions"
    ),
    logistic=True,
)

print_hypothesis_terms(
    return_model,
    (
        "Biased state -> engaged state "
        "within the next 3 trials"
    ),
    logistic=True,
)


# ---------------------------------------------------------------------
# 13. Descriptive plot using subject-balanced quintile summaries
#
# Each subject contributes one mean per feedback × pupil-bin condition,
# preventing high-trial-count animals from dominating the plot.
# ---------------------------------------------------------------------

plot_df = (
    analysis_df
    .dropna(
        subset=[
            "subject",
            "pupil_phasic_z",
            "negative_feedback",
            "future_js_max_3",
            "future_switch_any_3",
        ]
    )
    .copy()
)

plot_df["pupil_bin"] = pd.qcut(
    plot_df["pupil_phasic_z"],
    q=5,
    labels=[
        "Lowest",
        "Low",
        "Middle",
        "High",
        "Highest",
    ],
    duplicates="drop",
)

subject_bin_summary = (
    plot_df
    .groupby(
        [
            "subject",
            "negative_feedback",
            "pupil_bin",
        ],
        observed=True,
    )
    .agg(
        future_js=(
            "future_js_max_3",
            "mean",
        ),
        switch_probability=(
            "future_switch_any_3",
            "mean",
        ),
    )
    .reset_index()
)

population_bin_summary = (
    subject_bin_summary
    .groupby(
        [
            "negative_feedback",
            "pupil_bin",
        ],
        observed=True,
    )
    .agg(
        future_js_mean=(
            "future_js",
            "mean",
        ),
        future_js_sem=(
            "future_js",
            "sem",
        ),
        switch_mean=(
            "switch_probability",
            "mean",
        ),
        switch_sem=(
            "switch_probability",
            "sem",
        ),
        n_subjects=(
            "subject",
            "nunique",
        ),
    )
    .reset_index()
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
)

x = np.arange(5)

for feedback_value, label in [
    (0.0, "Rewarded"),
    (1.0, "Negative feedback"),
]:
    sub = (
        population_bin_summary.loc[
            population_bin_summary[
                "negative_feedback"
            ]
            == feedback_value
        ]
        .sort_values("pupil_bin")
    )

    axes[0].errorbar(
        x,
        sub["future_js_mean"],
        yerr=sub["future_js_sem"],
        marker="o",
        capsize=3,
        label=label,
    )

    axes[1].errorbar(
        x,
        sub["switch_mean"],
        yerr=sub["switch_sem"],
        marker="o",
        capsize=3,
        label=label,
    )


bin_labels = [
    "Lowest",
    "Low",
    "Middle",
    "High",
    "Highest",
]

for axis in axes:
    axis.set_xticks(x)
    axis.set_xticklabels(
        bin_labels,
        rotation=25,
        ha="right",
    )

    axis.set_xlabel(
        "Within-animal phasic pupil quintile"
    )

    axis.legend()


axes[0].set_ylabel(
    "Maximum posterior JSD\n"
    "over next 3 transitions"
)

axes[0].set_title(
    "Subsequent state-belief movement"
)

axes[1].set_ylabel(
    "Probability of any hard-state switch\n"
    "over next 3 transitions"
)

axes[1].set_title(
    "Subsequent inferred state switching"
)

plt.tight_layout()
plt.show()



# ===== CELL 53 =====
# =====================================================================
# EVENT-TRIGGERED JSD AROUND PHASIC PUPIL BURSTS
#
# Trial 0:
#   trial containing an unusually large phasic pupil response
#
# js_to_next at relative trial 0:
#   posterior change from the burst trial to the following trial
#
# Produces:
#   1. Overall event-triggered JSD
#   2. Separate curves for rewarded and negative-feedback burst trials
#
# Requires:
#   lability_df
#   pupil_phasic_clean or pupil_phasic
#   pupil_phasic_z
#   js_to_next
#   subject
#   sequence_id
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

PRE_TRIALS = 10
POST_TRIALS = 10

# Top 10% of each animal's phasic pupil distribution.
BURST_QUANTILE = 0.90

# Do not accept another burst within this many trials.
REFRACTORY_TRIALS = 5

# Require this many valid aligned observations from an animal
# before retaining its event-triggered curve.
MIN_VALID_OFFSETS_PER_SUBJECT = 8


# ---------------------------------------------------------------------
# 1. Prepare source table
# ---------------------------------------------------------------------

burst_df = lability_df.copy()

burst_df = burst_df.loc[
    burst_df["subject"].astype(str) != "SH015"
].copy()

required_columns = [
    "subject",
    "sequence_id",
    "js_to_next",
]

missing_columns = [
    column
    for column in required_columns
    if column not in burst_df.columns
]

if missing_columns:
    raise KeyError(
        f"Missing required columns: {missing_columns}"
    )


# Use the cleaned phasic pupil measurement.
if "pupil_phasic_clean" in burst_df.columns:
    burst_df["phasic_for_burst"] = pd.to_numeric(
        burst_df["pupil_phasic_clean"],
        errors="coerce",
    )

elif "pupil_phasic" in burst_df.columns:
    burst_df["phasic_for_burst"] = pd.to_numeric(
        burst_df["pupil_phasic"],
        errors="coerce",
    )

else:
    raise KeyError(
        "Neither pupil_phasic_clean nor pupil_phasic exists."
    )


burst_df["js_to_next"] = pd.to_numeric(
    burst_df["js_to_next"],
    errors="coerce",
)


# ---------------------------------------------------------------------
# 2. Define negative feedback
# ---------------------------------------------------------------------

if "negative_feedback" not in burst_df.columns:
    if "feedbackType" not in burst_df.columns:
        raise KeyError(
            "feedbackType or negative_feedback is required."
        )

    feedback = pd.to_numeric(
        burst_df["feedbackType"],
        errors="coerce",
    )

    burst_df["negative_feedback"] = np.where(
        feedback.notna(),
        feedback.ne(1).astype(float),
        np.nan,
    )


# ---------------------------------------------------------------------
# 3. Compute within-animal burst thresholds
#
# The threshold is animal-specific because pupil scale and variance
# differ substantially between animals.
# ---------------------------------------------------------------------

burst_threshold = (
    burst_df
    .groupby(
        "subject",
        sort=False,
    )["phasic_for_burst"]
    .transform(
        lambda values: values.quantile(
            BURST_QUANTILE
        )
    )
)

burst_df["above_burst_threshold"] = (
    burst_df["phasic_for_burst"]
    >= burst_threshold
)


# ---------------------------------------------------------------------
# 4. Identify local maxima within each recording sequence
# ---------------------------------------------------------------------

sequence_columns = [
    "subject",
    "sequence_id",
]

previous_phasic = (
    burst_df
    .groupby(
        sequence_columns,
        sort=False,
    )["phasic_for_burst"]
    .shift(1)
)

next_phasic = (
    burst_df
    .groupby(
        sequence_columns,
        sort=False,
    )["phasic_for_burst"]
    .shift(-1)
)

burst_df["is_local_maximum"] = (
    burst_df["phasic_for_burst"].notna()
    & previous_phasic.notna()
    & next_phasic.notna()
    & burst_df["phasic_for_burst"].gt(
        previous_phasic
    )
    & burst_df["phasic_for_burst"].ge(
        next_phasic
    )
)

burst_df["burst_candidate"] = (
    burst_df["above_burst_threshold"]
    & burst_df["is_local_maximum"]
)


# ---------------------------------------------------------------------
# 5. Apply an isolation/refractory criterion
#
# Within each sequence, retain the largest candidate when multiple
# candidates occur within the refractory window.
# ---------------------------------------------------------------------

burst_df["is_phasic_burst"] = False


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


for _, sequence in burst_df.groupby(
    sequence_columns,
    sort=False,
):
    selected_positions = select_isolated_bursts(
        sequence
    )

    if not selected_positions:
        continue

    selected_indices = (
        sequence.iloc[
            selected_positions
        ].index
    )

    burst_df.loc[
        selected_indices,
        "is_phasic_burst",
    ] = True


# Give each burst a unique identifier.
burst_df["burst_id"] = pd.NA

burst_indices = burst_df.index[
    burst_df["is_phasic_burst"]
]

burst_df.loc[
    burst_indices,
    "burst_id",
] = np.arange(
    len(burst_indices)
)

burst_df["burst_id"] = pd.array(
    burst_df["burst_id"],
    dtype="Int64",
)


# ---------------------------------------------------------------------
# 6. Extract JSD windows around each burst
#
# Windows never cross subject or recording-sequence boundaries.
# ---------------------------------------------------------------------

aligned_rows = []

relative_trials = np.arange(
    -PRE_TRIALS,
    POST_TRIALS + 1,
)

for (
    subject,
    sequence_id,
), sequence in burst_df.groupby(
    sequence_columns,
    sort=False,
):
    sequence = sequence.copy()

    sequence_indices = sequence.index.to_numpy()

    burst_positions = np.flatnonzero(
        sequence["is_phasic_burst"]
        .to_numpy(dtype=bool)
    )

    for burst_position in burst_positions:
        burst_row = sequence.iloc[
            burst_position
        ]

        burst_id = int(
            burst_row["burst_id"]
        )

        burst_feedback = (
            burst_row["negative_feedback"]
        )

        burst_amplitude = (
            burst_row["phasic_for_burst"]
        )

        for relative_trial in relative_trials:
            target_position = (
                burst_position
                + relative_trial
            )

            if (
                target_position < 0
                or target_position >= len(sequence)
            ):
                continue

            target_row = sequence.iloc[
                target_position
            ]

            aligned_rows.append(
                {
                    "subject": str(subject),
                    "sequence_id": sequence_id,
                    "burst_id": burst_id,
                    "relative_trial": relative_trial,
                    "js_to_next": target_row[
                        "js_to_next"
                    ],
                    "burst_negative_feedback": (
                        burst_feedback
                    ),
                    "burst_phasic_amplitude": (
                        burst_amplitude
                    ),
                }
            )


aligned_js = pd.DataFrame(
    aligned_rows
)

if aligned_js.empty:
    raise ValueError(
        "No complete pupil-burst windows were extracted."
    )


# ---------------------------------------------------------------------
# 7. Average events within each animal first
#
# This prevents animals with many detected bursts from dominating.
# ---------------------------------------------------------------------

subject_aligned_js = (
    aligned_js
    .dropna(
        subset=[
            "js_to_next",
            "burst_negative_feedback",
        ]
    )
    .groupby(
        [
            "subject",
            "relative_trial",
        ],
        as_index=False,
    )
    .agg(
        mean_js=(
            "js_to_next",
            "mean",
        ),
        n_bursts=(
            "burst_id",
            "nunique",
        ),
    )
)


subject_feedback_aligned_js = (
    aligned_js
    .dropna(
        subset=[
            "js_to_next",
            "burst_negative_feedback",
        ]
    )
    .groupby(
        [
            "subject",
            "burst_negative_feedback",
            "relative_trial",
        ],
        as_index=False,
    )
    .agg(
        mean_js=(
            "js_to_next",
            "mean",
        ),
        n_bursts=(
            "burst_id",
            "nunique",
        ),
    )
)


# ---------------------------------------------------------------------
# 8. Population mean and between-animal SEM
# ---------------------------------------------------------------------

population_js = (
    subject_aligned_js
    .groupby(
        "relative_trial",
        as_index=False,
    )
    .agg(
        mean_js=(
            "mean_js",
            "mean",
        ),
        sem_js=(
            "mean_js",
            "sem",
        ),
        n_subjects=(
            "subject",
            "nunique",
        ),
    )
)


population_feedback_js = (
    subject_feedback_aligned_js
    .groupby(
        [
            "burst_negative_feedback",
            "relative_trial",
        ],
        as_index=False,
    )
    .agg(
        mean_js=(
            "mean_js",
            "mean",
        ),
        sem_js=(
            "mean_js",
            "sem",
        ),
        n_subjects=(
            "subject",
            "nunique",
        ),
    )
)


# ---------------------------------------------------------------------
# 9. Event counts
# ---------------------------------------------------------------------

burst_summary = (
    burst_df.loc[
        burst_df["is_phasic_burst"],
        [
            "subject",
            "burst_id",
            "negative_feedback",
        ],
    ]
    .dropna(
        subset=[
            "burst_id",
            "negative_feedback",
        ]
    )
)

print(
    "Detected pupil bursts:",
    f"{burst_summary['burst_id'].nunique():,}",
)

print(
    "Animals contributing bursts:",
    burst_summary[
        "subject"
    ].nunique(),
)

print(
    "\nBurst trials by feedback:"
)

print(
    burst_summary[
        "negative_feedback"
    ]
    .value_counts()
    .rename(
        index={
            0.0: "Rewarded",
            1.0: "Negative feedback",
        }
    )
)


# ---------------------------------------------------------------------
# 10. Plot
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8),
    sharey=True,
)


# Overall pupil-burst alignment.
x = population_js[
    "relative_trial"
].to_numpy(dtype=float)

y = population_js[
    "mean_js"
].to_numpy(dtype=float)

sem = population_js[
    "sem_js"
].to_numpy(dtype=float)

axes[0].plot(
    x,
    y,
    marker="o",
    markersize=3,
    linewidth=1.8,
)

axes[0].fill_between(
    x,
    y - sem,
    y + sem,
    alpha=0.2,
)

axes[0].axvline(
    0,
    linestyle="--",
    linewidth=1,
    color="black",
)

axes[0].set_title(
    "State-belief movement around phasic pupil bursts"
)

axes[0].set_xlabel(
    "Trials relative to pupil burst"
)

axes[0].set_ylabel(
    "Jensen–Shannon divergence\n"
    "from current to next trial"
)


# Split by feedback on the burst trial.
for feedback_value, label in [
    (0.0, "Rewarded burst trial"),
    (1.0, "Negative-feedback burst trial"),
]:
    condition = (
        population_feedback_js.loc[
            population_feedback_js[
                "burst_negative_feedback"
            ]
            == feedback_value
        ]
        .sort_values(
            "relative_trial"
        )
    )

    x_condition = condition[
        "relative_trial"
    ].to_numpy(dtype=float)

    y_condition = condition[
        "mean_js"
    ].to_numpy(dtype=float)

    sem_condition = condition[
        "sem_js"
    ].to_numpy(dtype=float)

    axes[1].plot(
        x_condition,
        y_condition,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label=label,
    )

    axes[1].fill_between(
        x_condition,
        y_condition - sem_condition,
        y_condition + sem_condition,
        alpha=0.18,
    )


axes[1].axvline(
    0,
    linestyle="--",
    linewidth=1,
    color="black",
)

axes[1].set_title(
    "Burst-triggered lability by trial outcome"
)

axes[1].set_xlabel(
    "Trials relative to pupil burst"
)

axes[1].legend(
    frameon=False
)


for axis in axes:
    axis.axhline(
        0,
        linewidth=0.7,
        color="gray",
        alpha=0.5,
    )

    axis.set_xticks(
        np.arange(
            -PRE_TRIALS,
            POST_TRIALS + 1,
            2,
        )
    )


fig.suptitle(
    "GLM-HMM posterior lability time-locked to large phasic pupil responses",
    y=1.02,
)

plt.tight_layout()

plt.savefig(
    "output/jsd_timelocked_to_phasic_pupil_bursts.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()



# ===== CELL 54 =====
# =====================================================================
# BURST COVERAGE AND NEGATIVE-FEEDBACK ENRICHMENT DIAGNOSTICS
# =====================================================================

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Burst counts per animal and feedback condition
# ---------------------------------------------------------------------

subject_burst_counts = (
    burst_df.loc[
        burst_df["is_phasic_burst"]
        & burst_df["negative_feedback"].notna()
    ]
    .groupby(
        [
            "subject",
            "negative_feedback",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .rename(
        columns={
            0.0: "rewarded_bursts",
            1.0: "negative_feedback_bursts",
        }
    )
)

for column in [
    "rewarded_bursts",
    "negative_feedback_bursts",
]:
    if column not in subject_burst_counts.columns:
        subject_burst_counts[column] = 0


print("Burst counts per animal:")
print(
    subject_burst_counts[
        [
            "rewarded_bursts",
            "negative_feedback_bursts",
        ]
    ]
    .describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
    .round(1)
)


print("\nAnimals contributing at least one burst:")

print(
    "  Rewarded:",
    int(
        (
            subject_burst_counts[
                "rewarded_bursts"
            ]
            > 0
        ).sum()
    ),
)

print(
    "  Negative feedback:",
    int(
        (
            subject_burst_counts[
                "negative_feedback_bursts"
            ]
            > 0
        ).sum()
    ),
)


for minimum_count in [
    3,
    5,
    10,
]:
    n_both = (
        (
            subject_burst_counts[
                "rewarded_bursts"
            ]
            >= minimum_count
        )
        & (
            subject_burst_counts[
                "negative_feedback_bursts"
            ]
            >= minimum_count
        )
    ).sum()

    print(
        f"  At least {minimum_count} bursts "
        f"in both conditions: {n_both}"
    )


# ---------------------------------------------------------------------
# 2. Compare feedback rate on burst trials with the overall rate
#
# Restrict the baseline to trials that could have entered the analysis:
# clean pupil, feedback available, and valid next-trial JSD.
# ---------------------------------------------------------------------

eligible_trials = burst_df.loc[
    burst_df["phasic_for_burst"].notna()
    & burst_df["negative_feedback"].notna()
    & burst_df["js_to_next"].notna()
].copy()

burst_trials = eligible_trials.loc[
    eligible_trials["is_phasic_burst"]
].copy()


overall_failure_rate = (
    eligible_trials[
        "negative_feedback"
    ].mean()
)

burst_failure_rate = (
    burst_trials[
        "negative_feedback"
    ].mean()
)

risk_ratio = (
    burst_failure_rate
    / overall_failure_rate
    if overall_failure_rate > 0
    else np.nan
)

overall_odds = (
    overall_failure_rate
    / (
        1 - overall_failure_rate
    )
)

burst_odds = (
    burst_failure_rate
    / (
        1 - burst_failure_rate
    )
)

odds_ratio = (
    burst_odds
    / overall_odds
    if overall_odds > 0
    else np.nan
)


print("\nFeedback composition:")

print(
    "  Negative-feedback rate across all eligible trials:",
    f"{overall_failure_rate:.3%}",
)

print(
    "  Negative-feedback rate on pupil-burst trials:",
    f"{burst_failure_rate:.3%}",
)

print(
    "  Descriptive risk ratio:",
    f"{risk_ratio:.3f}",
)

print(
    "  Descriptive odds ratio:",
    f"{odds_ratio:.3f}",
)


# ---------------------------------------------------------------------
# 3. Subject-level comparison
#
# This avoids letting animals with more trials dominate the estimate.
# ---------------------------------------------------------------------

subject_rates = (
    eligible_trials
    .groupby(
        "subject"
    )
    .apply(
        lambda subject_df: pd.Series(
            {
                "overall_failure_rate": (
                    subject_df[
                        "negative_feedback"
                    ].mean()
                ),
                "burst_failure_rate": (
                    subject_df.loc[
                        subject_df[
                            "is_phasic_burst"
                        ],
                        "negative_feedback",
                    ].mean()
                ),
                "n_bursts": int(
                    subject_df[
                        "is_phasic_burst"
                    ].sum()
                ),
            }
        ),
        include_groups=False,
    )
    .dropna()
)


print("\nSubject-balanced rates:")

print(
    "  Mean overall negative-feedback rate:",
    f"{subject_rates['overall_failure_rate'].mean():.3%}",
)

print(
    "  Mean negative-feedback rate on burst trials:",
    f"{subject_rates['burst_failure_rate'].mean():.3%}",
)

print(
    "  Mean within-animal difference:",
    f"{(
        subject_rates['burst_failure_rate']
        - subject_rates['overall_failure_rate']
    ).mean():.3%}",
)



# ===== CELL 55 =====
# =====================================================================
# BURST-TRIGGERED JSD WITH MATCHED NON-BURST PSEUDO-EVENTS
#
# Primary test:
#
#   delta_JSD =
#       mean(JSD at relative trials 0, +1, +2)
#       -
#       mean(JSD at relative trials -3, -2, -1)
#
# Matching variables:
#   subject
#   recording sequence
#   feedback condition
#   current GLM-HMM state
#   approximate position within recording sequence
#
# Real and pseudo-events are paired one-to-one.
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon


# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------

PRE_OFFSETS = np.array(
    [-3, -2, -1],
    dtype=int,
)

POST_OFFSETS = np.array(
    [0, 1, 2],
    dtype=int,
)

ALL_OFFSETS = np.concatenate(
    [
        PRE_OFFSETS,
        POST_OFFSETS,
    ]
)

# Number of bins used to match approximate recording position.
N_POSITION_BINS = 10

# Pseudo-events this close to any real burst are excluded.
BURST_EXCLUSION_RADIUS = 5

RANDOM_SEED = 2026

rng = np.random.default_rng(
    RANDOM_SEED
)


# ---------------------------------------------------------------------
# 1. Prepare the trial table
# ---------------------------------------------------------------------

if "burst_df" in globals():
    event_source = burst_df.copy()

elif "lability_df" in globals():
    event_source = lability_df.copy()

else:
    raise NameError(
        "Neither burst_df nor lability_df exists."
    )


# ---------------------------------------------------------------------
# Validate matching-table columns
# ---------------------------------------------------------------------

required_real_burst_columns = {
    "pair_id",
    "_row_id",
    "subject",
    "sequence_id",
    "negative_feedback",
    "hard_state",
    "position_bin",
    "trial_fraction",
}

required_pseudo_candidate_columns = {
    "_row_id",
    "subject",
    "sequence_id",
    "negative_feedback",
    "hard_state",
    "position_bin",
    "trial_fraction",
}


missing_real_columns = (
    required_real_burst_columns
    - set(real_bursts.columns)
)

if missing_real_columns:
    raise KeyError(
        "real_bursts is missing: "
        f"{sorted(missing_real_columns)}"
    )


missing_pseudo_columns = (
    required_pseudo_candidate_columns
    - set(pseudo_candidates.columns)
)

if missing_pseudo_columns:
    raise KeyError(
        "pseudo_candidates is missing: "
        f"{sorted(missing_pseudo_columns)}"
    )


# Matching identifiers should be unique in their respective tables.
if not real_bursts["pair_id"].is_unique:
    raise ValueError(
        "real_bursts contains duplicated pair_id values."
    )

if not pseudo_candidates["_row_id"].is_unique:
    raise ValueError(
        "pseudo_candidates contains duplicated _row_id values."
    )


print(
    "Validation passed:",
    f"{len(real_bursts):,} real bursts and "
    f"{len(pseudo_candidates):,} pseudo-event candidates."
)


# ---------------------------------------------------------------------
# 2. Define feedback condition
# ---------------------------------------------------------------------

if "negative_feedback" not in event_source.columns:
    if "feedbackType" not in event_source.columns:
        raise KeyError(
            "negative_feedback or feedbackType is required."
        )

    feedback = pd.to_numeric(
        event_source["feedbackType"],
        errors="coerce",
    )

    event_source["negative_feedback"] = np.where(
        feedback.notna(),
        feedback.ne(1).astype(float),
        np.nan,
    )

else:
    event_source["negative_feedback"] = pd.to_numeric(
        event_source["negative_feedback"],
        errors="coerce",
    )


# ---------------------------------------------------------------------
# 3. Establish within-sequence trial position
# ---------------------------------------------------------------------

sequence_columns = [
    "subject",
    "sequence_id",
]

event_source = event_source.reset_index(
    drop=True
)

event_source["_row_id"] = np.arange(
    len(event_source)
)

event_source["trial_in_sequence"] = (
    event_source
    .groupby(
        sequence_columns,
        sort=False,
    )
    .cumcount()
)

event_source["sequence_length"] = (
    event_source
    .groupby(
        sequence_columns,
        sort=False,
    )["_row_id"]
    .transform("size")
)

event_source["trial_fraction"] = np.where(
    event_source["sequence_length"] > 1,
    event_source["trial_in_sequence"]
    / (
        event_source["sequence_length"]
        - 1
    ),
    0.0,
)

event_source["position_bin"] = np.floor(
    event_source["trial_fraction"]
    * N_POSITION_BINS
).astype(int)

event_source["position_bin"] = (
    event_source["position_bin"]
    .clip(
        lower=0,
        upper=N_POSITION_BINS - 1,
    )
)


# ---------------------------------------------------------------------
# 4. Restrict to events with complete possible windows
#
# At offset +2, js_to_next represents the transition from +2 to +3.
# Therefore, trial +3 must still exist within the sequence.
# ---------------------------------------------------------------------

event_source["window_inside_sequence"] = (
    event_source["trial_in_sequence"]
    >= abs(PRE_OFFSETS.min())
) & (
    event_source["trial_in_sequence"]
    <= (
        event_source["sequence_length"]
        - POST_OFFSETS.max()
        - 2
    )
)


# ---------------------------------------------------------------------
# 5. Exclude pseudo-events close to actual pupil bursts
# ---------------------------------------------------------------------

event_source["near_real_burst"] = False

for _, sequence in event_source.groupby(
    sequence_columns,
    sort=False,
):
    sequence_positions = (
        sequence["trial_in_sequence"]
        .to_numpy(dtype=int)
    )

    burst_positions = (
        sequence.loc[
            sequence["is_phasic_burst"],
            "trial_in_sequence",
        ]
        .to_numpy(dtype=int)
    )

    if len(burst_positions) == 0:
        continue

    near_burst = np.zeros(
        len(sequence),
        dtype=bool,
    )

    for burst_position in burst_positions:
        near_burst |= (
            np.abs(
                sequence_positions
                - burst_position
            )
            <= BURST_EXCLUSION_RADIUS
        )

    event_source.loc[
        sequence.index,
        "near_real_burst",
    ] = near_burst


# ---------------------------------------------------------------------
# 6. Identify eligible real bursts and pseudo-event candidates
# ---------------------------------------------------------------------

common_validity = (
    event_source["window_inside_sequence"]
    & event_source["negative_feedback"].notna()
    & event_source["hard_state"].notna()
)

real_bursts = event_source.loc[
    common_validity
    & event_source["is_phasic_burst"]
].copy()

pseudo_candidates = event_source.loc[
    common_validity
    & ~event_source["is_phasic_burst"]
    & ~event_source["near_real_burst"]
].copy()


# Assign a unique pair ID to each real event.
real_bursts = real_bursts.reset_index(
    drop=True
)

real_bursts["pair_id"] = np.arange(
    len(real_bursts),
    dtype=int,
)


print(
    "Eligible real pupil bursts:",
    f"{len(real_bursts):,}",
)

print(
    "Eligible non-burst pseudo-event candidates:",
    f"{len(pseudo_candidates):,}",
)


# ---------------------------------------------------------------------
# 7. Match one pseudo-event to each real burst
#
# Matching is exact for:
#   subject
#   sequence
#   feedback
#   hard state
#
# Position bin is matched as closely as possible.
#
# Candidates are used without replacement whenever possible.
# ---------------------------------------------------------------------

b# =====================================================================
# STRICT ONE-TO-ONE MATCHING FOR NON-BURST PSEUDO-EVENTS
#
# Replaces the previous matching section.
#
# Guarantees:
#   - no pseudo-event reuse
#   - same subject
#   - same recording sequence
#   - same feedback condition
#   - same current GLM-HMM state
#   - same task block, when probabilityLeft is available
#   - position-bin difference <= 1
#   - normalized trial-position difference <= 0.10
#
# Bursts without an acceptable control are discarded.
# =====================================================================

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment


MAX_POSITION_BIN_DIFFERENCE = 1
MAX_TRIAL_FRACTION_DIFFERENCE = 0.10


# ---------------------------------------------------------------------
# 1. Matching strata
# ---------------------------------------------------------------------

match_columns = [
    "subject",
    "sequence_id",
    "negative_feedback",
    "hard_state",
]

# Matching task block as well is preferable when available.
if (
    "probabilityLeft" in real_bursts.columns
    and "probabilityLeft" in pseudo_candidates.columns
):
    match_columns.append(
        "probabilityLeft"
    )


print(
    "Exact matching columns:",
    match_columns,
)


# ---------------------------------------------------------------------
# 2. Build pseudo-event pools
# ---------------------------------------------------------------------

pseudo_groups = {
    key: group.copy()
    for key, group in pseudo_candidates.groupby(
        match_columns,
        sort=False,
        dropna=False,
    )
}


# ---------------------------------------------------------------------
# 3. Optimal one-to-one matching within each stratum
#
# Hungarian assignment minimizes total trial-position distance.
# Invalid pairs receive a prohibitive cost and are removed afterward.
# ---------------------------------------------------------------------

strict_match_rows = []

LARGE_INVALID_COST = 1e6


for group_key, burst_group in real_bursts.groupby(
    match_columns,
    sort=False,
    dropna=False,
):
    control_group = pseudo_groups.get(
        group_key
    )

    if (
        control_group is None
        or control_group.empty
        or burst_group.empty
    ):
        continue

    burst_group = burst_group.copy()
    control_group = control_group.copy()

    burst_fraction = (
        burst_group["trial_fraction"]
        .to_numpy(dtype=float)
    )

    control_fraction = (
        control_group["trial_fraction"]
        .to_numpy(dtype=float)
    )

    burst_bins = (
        burst_group["position_bin"]
        .to_numpy(dtype=int)
    )

    control_bins = (
        control_group["position_bin"]
        .to_numpy(dtype=int)
    )

    # Rows correspond to real bursts.
    # Columns correspond to possible pseudo-events.
    fraction_distance = np.abs(
        burst_fraction[:, None]
        - control_fraction[None, :]
    )

    bin_distance = np.abs(
        burst_bins[:, None]
        - control_bins[None, :]
    )

    valid_pair = (
        fraction_distance
        <= MAX_TRIAL_FRACTION_DIFFERENCE
    ) & (
        bin_distance
        <= MAX_POSITION_BIN_DIFFERENCE
    )

    cost_matrix = np.where(
        valid_pair,
        fraction_distance,
        LARGE_INVALID_COST,
    )

    burst_indices, control_indices = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    for burst_index, control_index in zip(
        burst_indices,
        control_indices,
    ):
        if not valid_pair[
            burst_index,
            control_index,
        ]:
            continue

        burst_row = burst_group.iloc[
            burst_index
        ]

        control_row = control_group.iloc[
            control_index
        ]

        strict_match_rows.append(
            {
                "pair_id": int(
                    burst_row["pair_id"]
                ),
                "pseudo_row_id": int(
                    control_row["_row_id"]
                ),
                "burst_position_bin": int(
                    burst_row["position_bin"]
                ),
                "pseudo_position_bin": int(
                    control_row["position_bin"]
                ),
                "position_bin_difference": int(
                    abs(
                        burst_row["position_bin"]
                        - control_row["position_bin"]
                    )
                ),
                "trial_fraction_difference": float(
                    abs(
                        burst_row["trial_fraction"]
                        - control_row["trial_fraction"]
                    )
                ),
            }
        )


matches = pd.DataFrame(
    strict_match_rows
)


if matches.empty:
    raise ValueError(
        "No strict burst–control matches were found."
    )


# ---------------------------------------------------------------------
# 4. Validate matching
# ---------------------------------------------------------------------

assert matches["pair_id"].is_unique, (
    "A real burst was matched more than once."
)

assert matches["pseudo_row_id"].is_unique, (
    "A pseudo-event was reused."
)

assert (
    matches["position_bin_difference"]
    <= MAX_POSITION_BIN_DIFFERENCE
).all(), (
    "A match exceeded the allowed position-bin difference."
)

assert (
    matches["trial_fraction_difference"]
    <= MAX_TRIAL_FRACTION_DIFFERENCE
).all(), (
    "A match exceeded the allowed sequence-position distance."
)


n_unmatched = (
    len(real_bursts)
    - len(matches)
)


print(
    "\nStrict matched burst–pseudo pairs:",
    f"{len(matches):,}",
)

print(
    "Unmatched bursts discarded:",
    f"{n_unmatched:,}",
)

print(
    "Percentage of eligible bursts retained:",
    f"{len(matches) / len(real_bursts):.2%}",
)

print(
    "Exact position-bin matches:",
    f"{matches['position_bin_difference'].eq(0).mean():.2%}",
)

print(
    "Maximum trial-fraction difference:",
    f"{matches['trial_fraction_difference'].max():.4f}",
)

print(
    "Median trial-fraction difference:",
    f"{matches['trial_fraction_difference'].median():.4f}",
)

print(
    "Pseudo-event reuse count:",
    int(
        matches["pseudo_row_id"]
        .duplicated()
        .sum()
    ),
)


# ---------------------------------------------------------------------
# 5. Recreate observed-event table
# ---------------------------------------------------------------------

matched_real_events = (
    real_bursts
    .merge(
        matches,
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    .rename(
        columns={
            "trial_in_sequence":
                "event_trial_position",
        }
    )
)

matched_real_events["event_type"] = (
    "Observed pupil burst"
)

matched_real_events = matched_real_events[
    [
        "pair_id",
        "subject",
        "sequence_id",
        "event_trial_position",
        "negative_feedback",
        "hard_state",
        "event_type",
    ]
]


# ---------------------------------------------------------------------
# 6. Recreate pseudo-event table
# ---------------------------------------------------------------------

pseudo_lookup = (
    event_source[
        [
            "_row_id",
            "subject",
            "sequence_id",
            "trial_in_sequence",
            "negative_feedback",
            "hard_state",
        ]
    ]
    .rename(
        columns={
            "_row_id":
                "pseudo_row_id",
            "trial_in_sequence":
                "event_trial_position",
        }
    )
)

assert pseudo_lookup[
    "pseudo_row_id"
].is_unique


matched_pseudo_events = (
    matches
    .merge(
        pseudo_lookup,
        on="pseudo_row_id",
        how="left",
        validate="one_to_one",
    )
)

assert matched_pseudo_events[
    "pair_id"
].is_unique

assert matched_pseudo_events[
    "subject"
].notna().all()


matched_pseudo_events["event_type"] = (
    "Matched non-burst"
)

matched_pseudo_events = matched_pseudo_events[
    [
        "pair_id",
        "subject",
        "sequence_id",
        "event_trial_position",
        "negative_feedback",
        "hard_state",
        "event_type",
    ]
]


event_table = pd.concat(
    [
        matched_real_events,
        matched_pseudo_events,
    ],
    ignore_index=True,
)


# Exactly two rows should exist per pair:
# one observed burst and one matched control.
pair_counts = (
    event_table
    .groupby(
        "pair_id"
    )
    .size()
)

assert pair_counts.eq(2).all(), (
    "Each pair must contain one burst and one control."
)


print(
    "\nStrict event table created successfully."
)

print(
    "Observed events:",
    int(
        (
            event_table["event_type"]
            == "Observed pupil burst"
        ).sum()
    ),
)

print(
    "Matched controls:",
    int(
        (
            event_table["event_type"]
            == "Matched non-burst"
        ).sum()
    ),
)


# ---------------------------------------------------------------------
# 8. Create real and pseudo-event tables
# ---------------------------------------------------------------------

# Every real burst should have one unique pair_id.
assert matches["pair_id"].is_unique, (
    "pair_id should be unique in matches."
)

matched_real_events = (
    real_bursts
    .merge(
        matches,
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    .rename(
        columns={
            "trial_in_sequence":
                "event_trial_position",
        }
    )
)

matched_real_events["event_type"] = (
    "Observed pupil burst"
)

matched_real_events = matched_real_events[
    [
        "pair_id",
        "subject",
        "sequence_id",
        "event_trial_position",
        "negative_feedback",
        "hard_state",
        "event_type",
    ]
]


# One row per trial in the source table.
pseudo_lookup = (
    event_source[
        [
            "_row_id",
            "subject",
            "sequence_id",
            "trial_in_sequence",
            "negative_feedback",
            "hard_state",
        ]
    ]
    .rename(
        columns={
            "_row_id": "pseudo_row_id",
            "trial_in_sequence":
                "event_trial_position",
        }
    )
)

assert pseudo_lookup["pseudo_row_id"].is_unique, (
    "pseudo_lookup should contain one row per pseudo_row_id."
)


# Multiple matched bursts may reference the same pseudo-event because
# the fallback matching procedure permits sampling with replacement.
matched_pseudo_events = (
    matches
    .merge(
        pseudo_lookup,
        on="pseudo_row_id",
        how="left",
        validate="many_to_one",
    )
)

# The pair identifier should remain unique even when pseudo-events
# themselves are reused.
assert matched_pseudo_events["pair_id"].is_unique, (
    "Each burst–pseudo pair should still have a unique pair_id."
)

assert matched_pseudo_events["subject"].notna().all(), (
    "Some matched pseudo_row_id values were not found in event_source."
)

matched_pseudo_events["event_type"] = (
    "Matched non-burst"
)

matched_pseudo_events = matched_pseudo_events[
    [
        "pair_id",
        "subject",
        "sequence_id",
        "event_trial_position",
        "negative_feedback",
        "hard_state",
        "event_type",
    ]
]


event_table = pd.concat(
    [
        matched_real_events,
        matched_pseudo_events,
    ],
    ignore_index=True,
)


# ---------------------------------------------------------------------
# Matching diagnostics
# ---------------------------------------------------------------------

n_reused_rows = (
    matches["pseudo_row_id"]
    .duplicated(
        keep=False
    )
    .sum()
)

n_unique_pseudo = (
    matches["pseudo_row_id"]
    .nunique()
)

reuse_fraction = (
    1
    - n_unique_pseudo / len(matches)
)

print(
    "Matched event pairs:",
    f"{len(matches):,}",
)

print(
    "Unique pseudo-events used:",
    f"{n_unique_pseudo:,}",
)

print(
    "Rows involving a reused pseudo-event:",
    f"{n_reused_rows:,}",
)

print(
    "Fraction of matches attributable to reuse:",
    f"{reuse_fraction:.2%}",
)

# ---------------------------------------------------------------------
# 9. Extract aligned JSD windows using a vectorized merge
# ---------------------------------------------------------------------

offset_table = pd.DataFrame(
    {
        "relative_trial":
            ALL_OFFSETS
    }
)

event_table["_merge_key"] = 1
offset_table["_merge_key"] = 1

aligned_events = (
    event_table
    .merge(
        offset_table,
        on="_merge_key",
        how="inner",
    )
    .drop(
        columns="_merge_key"
    )
)

aligned_events[
    "target_trial_position"
] = (
    aligned_events[
        "event_trial_position"
    ]
    + aligned_events[
        "relative_trial"
    ]
)


trial_js_lookup = event_source[
    [
        "subject",
        "sequence_id",
        "trial_in_sequence",
        "js_to_next",
    ]
].rename(
    columns={
        "trial_in_sequence":
            "target_trial_position",
    }
)


aligned_events = aligned_events.merge(
    trial_js_lookup,
    on=[
        "subject",
        "sequence_id",
        "target_trial_position",
    ],
    how="left",
    validate="many_to_one",
)


# ---------------------------------------------------------------------
# 10. Calculate event-level pre, post, and delta JSD
# ---------------------------------------------------------------------

aligned_events["period"] = np.where(
    aligned_events[
        "relative_trial"
    ].isin(
        PRE_OFFSETS
    ),
    "pre",
    "post",
)


event_period_summary = (
    aligned_events
    .groupby(
        [
            "pair_id",
            "subject",
            "negative_feedback",
            "hard_state",
            "event_type",
            "period",
        ],
        as_index=False,
    )
    .agg(
        mean_js=(
            "js_to_next",
            "mean",
        ),
        n_valid=(
            "js_to_next",
            "count",
        ),
    )
)


event_period_wide = (
    event_period_summary
    .pivot_table(
        index=[
            "pair_id",
            "subject",
            "negative_feedback",
            "hard_state",
            "event_type",
        ],
        columns="period",
        values=[
            "mean_js",
            "n_valid",
        ],
        aggfunc="first",
    )
)

event_period_wide.columns = [
    f"{value}_{period}"
    for value, period
    in event_period_wide.columns
]

event_delta = (
    event_period_wide
    .reset_index()
)


# Require all three pre and all three post JSD observations.
event_delta = event_delta.loc[
    event_delta["n_valid_pre"].eq(
        len(PRE_OFFSETS)
    )
    & event_delta["n_valid_post"].eq(
        len(POST_OFFSETS)
    )
].copy()


event_delta["delta_js"] = (
    event_delta["mean_js_post"]
    - event_delta["mean_js_pre"]
)


event_delta["feedback_label"] = np.where(
    event_delta["negative_feedback"]
    == 1,
    "Negative feedback",
    "Rewarded",
)


print(
    "\nComplete event windows:",
    f"{len(event_delta):,}",
)


# ---------------------------------------------------------------------
# 11. Average within animal before statistical testing
# ---------------------------------------------------------------------

subject_delta = (
    event_delta
    .groupby(
        [
            "subject",
            "negative_feedback",
            "feedback_label",
            "event_type",
        ],
        as_index=False,
    )
    .agg(
        mean_delta_js=(
            "delta_js",
            "mean",
        ),
        n_events=(
            "pair_id",
            "nunique",
        ),
    )
)


population_delta = (
    subject_delta
    .groupby(
        [
            "negative_feedback",
            "feedback_label",
            "event_type",
        ],
        as_index=False,
    )
    .agg(
        mean_delta_js=(
            "mean_delta_js",
            "mean",
        ),
        sem_delta_js=(
            "mean_delta_js",
            "sem",
        ),
        n_subjects=(
            "subject",
            "nunique",
        ),
    )
)


print(
    "\nSubject-balanced ΔJSD:"
)

population_delta_display = (
    population_delta
    .sort_values(
        [
            "event_type",
            "negative_feedback",
        ]
    )
    [
        [
            "feedback_label",
            "event_type",
            "mean_delta_js",
            "sem_delta_js",
            "n_subjects",
        ]
    ]
)

print(
    population_delta_display.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.6f}"
        ),
    )
)


# ---------------------------------------------------------------------
# 12. Statistical tests
# ---------------------------------------------------------------------

def paired_wilcoxon(
    dataframe,
    index_column,
    condition_column,
    value_column,
    condition_a,
    condition_b,
    label,
):
    wide = (
        dataframe
        .pivot_table(
            index=index_column,
            columns=condition_column,
            values=value_column,
            aggfunc="first",
        )
        .dropna(
            subset=[
                condition_a,
                condition_b,
            ]
        )
    )

    difference = (
        wide[condition_a]
        - wide[condition_b]
    )

    if len(difference) < 3:
        print(
            f"\n{label}: insufficient paired subjects."
        )
        return None

    statistic, p_value = wilcoxon(
        difference
    )

    difference_sd = difference.std(
        ddof=1
    )

    dz = (
        difference.mean()
        / difference_sd
        if (
            np.isfinite(difference_sd)
            and difference_sd > 0
        )
        else np.nan
    )

    print(
        f"\n{label}"
    )

    print(
        f"  n = {len(difference)}"
    )

    print(
        f"  mean paired difference = "
        f"{difference.mean():.6f}"
    )

    print(
        f"  median paired difference = "
        f"{difference.median():.6f}"
    )

    print(
        f"  Wilcoxon W = "
        f"{statistic:.1f}"
    )

    print(
        f"  p = {p_value:.6g}"
    )

    print(
        f"  dz = {dz:.3f}"
    )

    return {
        "wide": wide,
        "difference": difference,
        "W": statistic,
        "p": p_value,
        "dz": dz,
    }


# --------------------------------------------------
# Primary test:
# observed negative-feedback ΔJSD > observed rewarded ΔJSD
# --------------------------------------------------

observed_subject_delta = (
    subject_delta.loc[
        subject_delta["event_type"]
        == "Observed pupil burst"
    ]
    .copy()
)

primary_test = paired_wilcoxon(
    dataframe=observed_subject_delta,
    index_column="subject",
    condition_column="feedback_label",
    value_column="mean_delta_js",
    condition_a="Negative feedback",
    condition_b="Rewarded",
    label=(
        "Primary comparison: observed burst ΔJSD, "
        "negative feedback minus rewarded"
    ),
)


# --------------------------------------------------
# Observed burst versus matched pseudo-event
# within each feedback condition
# --------------------------------------------------

for feedback_label in [
    "Rewarded",
    "Negative feedback",
]:
    condition_df = subject_delta.loc[
        subject_delta["feedback_label"]
        == feedback_label
    ].copy()

    paired_wilcoxon(
        dataframe=condition_df,
        index_column="subject",
        condition_column="event_type",
        value_column="mean_delta_js",
        condition_a="Observed pupil burst",
        condition_b="Matched non-burst",
        label=(
            f"{feedback_label}: observed burst "
            "minus matched non-burst ΔJSD"
        ),
    )


# --------------------------------------------------
# Difference-in-differences
#
# [(observed - pseudo) after negative feedback]
# -
# [(observed - pseudo) after reward]
# --------------------------------------------------

did_table = (
    subject_delta
    .pivot_table(
        index="subject",
        columns=[
            "feedback_label",
            "event_type",
        ],
        values="mean_delta_js",
        aggfunc="first",
    )
)


required_did_columns = [
    (
        "Negative feedback",
        "Observed pupil burst",
    ),
    (
        "Negative feedback",
        "Matched non-burst",
    ),
    (
        "Rewarded",
        "Observed pupil burst",
    ),
    (
        "Rewarded",
        "Matched non-burst",
    ),
]


did_complete = did_table.dropna(
    subset=required_did_columns
).copy()


did_complete[
    "negative_feedback_burst_effect"
] = (
    did_complete[
        (
            "Negative feedback",
            "Observed pupil burst",
        )
    ]
    - did_complete[
        (
            "Negative feedback",
            "Matched non-burst",
        )
    ]
)

did_complete[
    "rewarded_burst_effect"
] = (
    did_complete[
        (
            "Rewarded",
            "Observed pupil burst",
        )
    ]
    - did_complete[
        (
            "Rewarded",
            "Matched non-burst",
        )
    ]
)

did_complete[
    "difference_in_differences"
] = (
    did_complete[
        "negative_feedback_burst_effect"
    ]
    - did_complete[
        "rewarded_burst_effect"
    ]
)


if len(did_complete) >= 3:
    did_statistic, did_p = wilcoxon(
        did_complete[
            "difference_in_differences"
        ]
    )

    did_difference = did_complete[
        "difference_in_differences"
    ]

    did_sd = did_difference.std(
        ddof=1
    )

    did_dz = (
        did_difference.mean()
        / did_sd
        if (
            np.isfinite(did_sd)
            and did_sd > 0
        )
        else np.nan
    )

    print(
        "\nDifference-in-differences:"
    )

    print(
        "  [(burst − pseudo) negative feedback] "
        "− [(burst − pseudo) rewarded]"
    )

    print(
        f"  n = {len(did_complete)}"
    )

    print(
        f"  mean = "
        f"{did_difference.mean():.6f}"
    )

    print(
        f"  Wilcoxon W = "
        f"{did_statistic:.1f}"
    )

    print(
        f"  p = {did_p:.6g}"
    )

    print(
        f"  dz = {did_dz:.3f}"
    )


# ---------------------------------------------------------------------
# 13. Subject-balanced time-locked curves
# ---------------------------------------------------------------------

subject_timecourse = (
    aligned_events
    .dropna(
        subset=[
            "js_to_next",
        ]
    )
    .groupby(
        [
            "subject",
            "negative_feedback",
            "event_type",
            "relative_trial",
        ],
        as_index=False,
    )
    .agg(
        mean_js=(
            "js_to_next",
            "mean",
        ),
        n_events=(
            "pair_id",
            "nunique",
        ),
    )
)


population_timecourse = (
    subject_timecourse
    .groupby(
        [
            "negative_feedback",
            "event_type",
            "relative_trial",
        ],
        as_index=False,
    )
    .agg(
        mean_js=(
            "mean_js",
            "mean",
        ),
        sem_js=(
            "mean_js",
            "sem",
        ),
        n_subjects=(
            "subject",
            "nunique",
        ),
    )
)


# ---------------------------------------------------------------------
# 14. Plot time-locked observed and matched-control curves
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        12,
        4.8,
    ),
    sharey=True,
)


feedback_panels = [
    (
        0.0,
        "Rewarded burst trials",
    ),
    (
        1.0,
        "Negative-feedback burst trials",
    ),
]


for axis, (
    feedback_value,
    panel_title,
) in zip(
    axes,
    feedback_panels,
):
    panel_data = (
        population_timecourse.loc[
            population_timecourse[
                "negative_feedback"
            ]
            == feedback_value
        ]
        .copy()
    )

    for event_type, linestyle in [
        (
            "Observed pupil burst",
            "-",
        ),
        (
            "Matched non-burst",
            "--",
        ),
    ]:
        curve = (
            panel_data.loc[
                panel_data[
                    "event_type"
                ]
                == event_type
            ]
            .sort_values(
                "relative_trial"
            )
        )

        x = curve[
            "relative_trial"
        ].to_numpy(dtype=float)

        y = curve[
            "mean_js"
        ].to_numpy(dtype=float)

        sem = curve[
            "sem_js"
        ].to_numpy(dtype=float)

        axis.plot(
            x,
            y,
            marker="o",
            markersize=4,
            linewidth=2,
            linestyle=linestyle,
            label=event_type,
        )

        axis.fill_between(
            x,
            y - sem,
            y + sem,
            alpha=0.16,
        )

    axis.axvline(
        0,
        color="black",
        linestyle=":",
        linewidth=1.2,
    )

    axis.axvspan(
        -3,
        -1,
        alpha=0.06,
    )

    axis.axvspan(
        0,
        2,
        alpha=0.10,
    )

    axis.set_title(
        panel_title
    )

    axis.set_xlabel(
        "Trials relative to event"
    )

    axis.set_xticks(
        ALL_OFFSETS
    )

    axis.legend(
        frameon=False
    )


axes[0].set_ylabel(
    "Jensen–Shannon divergence\n"
    "from current to next trial"
)

fig.suptitle(
    "GLM-HMM state-belief movement around pupil bursts\n"
    "Observed bursts compared with matched non-burst pseudo-events",
    y=1.04,
)

plt.tight_layout()

plt.savefig(
    "output/jsd_burst_vs_matched_pseudoevents.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# ---------------------------------------------------------------------
# 15. Plot subject-balanced ΔJSD
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        7.5,
        5.2,
    )
)


x_locations = {
    (
        "Rewarded",
        "Matched non-burst",
    ): 0.0,
    (
        "Rewarded",
        "Observed pupil burst",
    ): 0.8,
    (
        "Negative feedback",
        "Matched non-burst",
    ): 2.0,
    (
        "Negative feedback",
        "Observed pupil burst",
    ): 2.8,
}


for (
    feedback_label,
    event_type,
), x_position in x_locations.items():

    values = subject_delta.loc[
        (
            subject_delta[
                "feedback_label"
            ]
            == feedback_label
        )
        & (
            subject_delta[
                "event_type"
            ]
            == event_type
        ),
        "mean_delta_js",
    ].dropna()

    jitter = rng.uniform(
        -0.10,
        0.10,
        size=len(values),
    )

    ax.scatter(
        np.full(
            len(values),
            x_position,
        )
        + jitter,
        values,
        alpha=0.30,
        s=18,
    )

    ax.errorbar(
        x_position,
        values.mean(),
        yerr=values.sem(),
        marker="o",
        markersize=8,
        capsize=4,
        linewidth=2,
    )


ax.axhline(
    0,
    color="black",
    linewidth=1,
    alpha=0.6,
)

ax.set_xticks(
    [
        0.4,
        2.4,
    ]
)

ax.set_xticklabels(
    [
        "Rewarded",
        "Negative feedback",
    ]
)

ax.set_ylabel(
    "ΔJSD\n"
    "mean post-event JSD − mean pre-event JSD"
)

ax.set_title(
    "Change in posterior lability around pupil bursts\n"
    "versus matched non-burst events"
)

plt.tight_layout()

plt.savefig(
    "output/delta_jsd_burst_vs_matched_pseudoevents.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()



# ===== CELL 56 =====
# =====================================================================
# DIAGNOSE THE EXTREME NEGATIVE-FEEDBACK ΔJSD SUBJECT
#
# Identifies:
#   - the animal producing the extreme point
#   - whether it is in the burst or matched-control condition
#   - how many events contributed
#   - whether one or two individual pseudo-events caused the subject mean
#
# Then reruns the three primary tests:
#   - with all animals
#   - excluding the extreme animal
#   - requiring >= 3, 5, or 10 events per condition
# =====================================================================

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


# ---------------------------------------------------------------------
# 1. Construct a subject-level negative-feedback diagnostic table
# ---------------------------------------------------------------------

negative_subject_delta = (
    subject_delta.loc[
        subject_delta["feedback_label"]
        == "Negative feedback"
    ]
    .copy()
)


observed_negative = (
    negative_subject_delta.loc[
        negative_subject_delta["event_type"]
        == "Observed pupil burst",
        [
            "subject",
            "mean_delta_js",
            "n_events",
        ],
    ]
    .set_index("subject")
    .rename(
        columns={
            "mean_delta_js":
                "observed_burst_delta",
            "n_events":
                "observed_burst_n",
        }
    )
)


control_negative = (
    negative_subject_delta.loc[
        negative_subject_delta["event_type"]
        == "Matched non-burst",
        [
            "subject",
            "mean_delta_js",
            "n_events",
        ],
    ]
    .set_index("subject")
    .rename(
        columns={
            "mean_delta_js":
                "matched_control_delta",
            "n_events":
                "matched_control_n",
        }
    )
)


negative_diagnostic = (
    observed_negative
    .join(
        control_negative,
        how="outer",
    )
)


negative_diagnostic[
    "burst_minus_control"
] = (
    negative_diagnostic[
        "observed_burst_delta"
    ]
    - negative_diagnostic[
        "matched_control_delta"
    ]
)


# ---------------------------------------------------------------------
# 2. Calculate robust z-scores using median and MAD
# ---------------------------------------------------------------------

def robust_zscore(values):
    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    median = values.median()

    mad = (
        values - median
    ).abs().median()

    robust_scale = (
        1.4826 * mad
    )

    if (
        not np.isfinite(robust_scale)
        or robust_scale == 0
    ):
        return pd.Series(
            np.nan,
            index=values.index,
        )

    return (
        values - median
    ) / robust_scale


negative_diagnostic[
    "control_robust_z"
] = robust_zscore(
    negative_diagnostic[
        "matched_control_delta"
    ]
)

negative_diagnostic[
    "observed_robust_z"
] = robust_zscore(
    negative_diagnostic[
        "observed_burst_delta"
    ]
)


print(
    "Largest negative-feedback matched-control values:"
)

print(
    negative_diagnostic
    .sort_values(
        "matched_control_delta",
        ascending=False,
    )
    .head(10)
    .round(6)
    .to_string()
)


print(
    "\nLargest absolute robust control outliers:"
)

print(
    negative_diagnostic
    .assign(
        absolute_control_robust_z=lambda df: (
            df["control_robust_z"].abs()
        )
    )
    .sort_values(
        "absolute_control_robust_z",
        ascending=False,
    )
    .head(10)
    .round(6)
    .to_string()
)


# ---------------------------------------------------------------------
# 3. Identify the most extreme matched-control subject
# ---------------------------------------------------------------------

extreme_subject = (
    negative_diagnostic[
        "matched_control_delta"
    ]
    .abs()
    .idxmax()
)

print(
    "\nMost extreme matched-control subject:",
    extreme_subject,
)

print(
    negative_diagnostic.loc[
        extreme_subject
    ]
    .round(6)
    .to_string()
)


# ---------------------------------------------------------------------
# 4. Inspect all event-level ΔJSD values for this animal
# ---------------------------------------------------------------------

extreme_event_details = (
    event_delta.loc[
        (
            event_delta["subject"]
            .astype(str)
            == str(extreme_subject)
        )
        & (
            event_delta["feedback_label"]
            == "Negative feedback"
        ),
        [
            "pair_id",
            "event_type",
            "mean_js_pre",
            "mean_js_post",
            "delta_js",
        ],
    ]
    .sort_values(
        [
            "event_type",
            "delta_js",
        ],
        ascending=[
            True,
            False,
        ],
    )
)


print(
    "\nEvent-level values for the extreme animal:"
)

print(
    extreme_event_details
    .round(6)
    .to_string(
        index=False
    )
)


print(
    "\nEvent-level summary for the extreme animal:"
)

print(
    extreme_event_details
    .groupby(
        "event_type"
    )["delta_js"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
    .round(6)
    .to_string()
)


# ---------------------------------------------------------------------
# 5. Show the individual matched-control events that contributed most
# ---------------------------------------------------------------------

extreme_control_events = (
    extreme_event_details.loc[
        extreme_event_details["event_type"]
        == "Matched non-burst"
    ]
    .sort_values(
        "delta_js",
        ascending=False,
    )
)


print(
    "\nLargest matched-control pseudo-events "
    "for the extreme animal:"
)

print(
    extreme_control_events
    .head(15)
    .round(6)
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# 6. Check whether a pseudo-event was reused for this subject
# ---------------------------------------------------------------------

if {
    "matches",
    "pseudo_row_id",
}.issubset(
    set(globals())
):
    pass


if (
    "matches" in globals()
    and "pseudo_row_id"
    in matches.columns
):
    extreme_pair_ids = (
        extreme_control_events[
            "pair_id"
        ]
        .to_numpy()
    )

    extreme_matches = (
        matches.loc[
            matches["pair_id"].isin(
                extreme_pair_ids
            ),
            [
                "pair_id",
                "pseudo_row_id",
                "burst_position_bin",
                "pseudo_position_bin",
            ],
        ]
        .copy()
    )

    reuse_counts = (
        matches[
            "pseudo_row_id"
        ]
        .value_counts()
    )

    extreme_matches[
        "times_pseudo_event_used"
    ] = (
        extreme_matches[
            "pseudo_row_id"
        ]
        .map(
            reuse_counts
        )
    )

    print(
        "\nMatching diagnostics for this animal:"
    )

    print(
        extreme_matches
        .sort_values(
            "times_pseudo_event_used",
            ascending=False,
        )
        .head(20)
        .to_string(
            index=False
        )
    )


# =====================================================================
# SENSITIVITY TESTS
# =====================================================================

def run_signed_rank(
    differences,
    label,
):
    differences = (
        pd.Series(
            differences
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if len(differences) < 3:
        return {
            "comparison": label,
            "n": len(differences),
            "mean": np.nan,
            "median": np.nan,
            "W": np.nan,
            "p": np.nan,
            "dz": np.nan,
        }

    if np.allclose(
        differences,
        0,
    ):
        statistic = 0.0
        p_value = 1.0
    else:
        statistic, p_value = wilcoxon(
            differences
        )

    difference_sd = (
        differences.std(
            ddof=1
        )
    )

    dz = (
        differences.mean()
        / difference_sd
        if (
            np.isfinite(
                difference_sd
            )
            and difference_sd > 0
        )
        else np.nan
    )

    return {
        "comparison": label,
        "n": len(differences),
        "mean": differences.mean(),
        "median": differences.median(),
        "W": statistic,
        "p": p_value,
        "dz": dz,
    }


def sensitivity_analysis(
    subject_delta_table,
    excluded_subjects=None,
    minimum_events=1,
    analysis_label="All animals",
):
    data = subject_delta_table.copy()

    data["subject"] = (
        data["subject"]
        .astype(str)
    )

    excluded_subjects = {
        str(subject)
        for subject in (
            excluded_subjects
            if excluded_subjects is not None
            else []
        )
    }

    if excluded_subjects:
        data = data.loc[
            ~data["subject"].isin(
                excluded_subjects
            )
        ].copy()

    # Require the minimum event count within every condition entering
    # a given comparison.
    data = data.loc[
        data["n_events"]
        >= minimum_events
    ].copy()

    results = []


    # -------------------------------------------------------------
    # A. Observed bursts:
    #    negative feedback minus rewarded
    # -------------------------------------------------------------

    observed = (
        data.loc[
            data["event_type"]
            == "Observed pupil burst"
        ]
        .pivot_table(
            index="subject",
            columns="feedback_label",
            values="mean_delta_js",
            aggfunc="first",
        )
        .dropna(
            subset=[
                "Negative feedback",
                "Rewarded",
            ]
        )
    )

    observed_difference = (
        observed[
            "Negative feedback"
        ]
        - observed[
            "Rewarded"
        ]
    )

    result = run_signed_rank(
        observed_difference,
        (
            f"{analysis_label}: "
            "observed negative-feedback "
            "minus rewarded"
        ),
    )

    results.append(
        result
    )


    # -------------------------------------------------------------
    # B. Negative feedback:
    #    observed burst minus matched control
    # -------------------------------------------------------------

    negative = (
        data.loc[
            data["feedback_label"]
            == "Negative feedback"
        ]
        .pivot_table(
            index="subject",
            columns="event_type",
            values="mean_delta_js",
            aggfunc="first",
        )
        .dropna(
            subset=[
                "Observed pupil burst",
                "Matched non-burst",
            ]
        )
    )

    negative_difference = (
        negative[
            "Observed pupil burst"
        ]
        - negative[
            "Matched non-burst"
        ]
    )

    result = run_signed_rank(
        negative_difference,
        (
            f"{analysis_label}: "
            "negative-feedback burst "
            "minus matched control"
        ),
    )

    results.append(
        result
    )


    # -------------------------------------------------------------
    # C. Difference-in-differences
    # -------------------------------------------------------------

    did = (
        data
        .pivot_table(
            index="subject",
            columns=[
                "feedback_label",
                "event_type",
            ],
            values="mean_delta_js",
            aggfunc="first",
        )
    )

    required_columns = [
        (
            "Negative feedback",
            "Observed pupil burst",
        ),
        (
            "Negative feedback",
            "Matched non-burst",
        ),
        (
            "Rewarded",
            "Observed pupil burst",
        ),
        (
            "Rewarded",
            "Matched non-burst",
        ),
    ]

    did = did.dropna(
        subset=required_columns
    )

    did_difference = (
        (
            did[
                (
                    "Negative feedback",
                    "Observed pupil burst",
                )
            ]
            - did[
                (
                    "Negative feedback",
                    "Matched non-burst",
                )
            ]
        )
        - (
            did[
                (
                    "Rewarded",
                    "Observed pupil burst",
                )
            ]
            - did[
                (
                    "Rewarded",
                    "Matched non-burst",
                )
            ]
        )
    )

    result = run_signed_rank(
        did_difference,
        (
            f"{analysis_label}: "
            "difference-in-differences"
        ),
    )

    results.append(
        result
    )

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------------------
# 7. Run all sensitivity specifications
# ---------------------------------------------------------------------

sensitivity_results = []


# Original analysis.
sensitivity_results.append(
    sensitivity_analysis(
        subject_delta_table=
            subject_delta,
        excluded_subjects=[],
        minimum_events=1,
        analysis_label="All animals",
    )
)


# Exclude only the identified extreme subject.
sensitivity_results.append(
    sensitivity_analysis(
        subject_delta_table=
            subject_delta,
        excluded_subjects=[
            extreme_subject,
        ],
        minimum_events=1,
        analysis_label=(
            "Extreme subject excluded"
        ),
    )
)


# Minimum event-count analyses.
for minimum_events in [
    3,
    5,
    10,
]:
    sensitivity_results.append(
        sensitivity_analysis(
            subject_delta_table=
                subject_delta,
            excluded_subjects=[],
            minimum_events=
                minimum_events,
            analysis_label=(
                f"At least {minimum_events} "
                "events per condition"
            ),
        )
    )


sensitivity_results = pd.concat(
    sensitivity_results,
    ignore_index=True,
)


print(
    "\n"
    + "=" * 100
)

print(
    "OUTLIER AND EVENT-COUNT SENSITIVITY RESULTS"
)

print(
    "=" * 100
)

print(
    sensitivity_results.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.6f}"
        ),
    )
)



# ===== CELL 57 =====
# =====================================================================
# ROBUSTNESS CHECKS FOR BURST-TRIGGERED ΔJSD
#
# Uses subject-level MEDIAN event ΔJSD instead of the mean.
#
# Matching specifications:
#   1. Exact position bin
#   2. Trial-position difference <= 5% of sequence
#   3. Exact position bin AND <= 5% difference
#
# Requires existing objects:
#   event_source
#   real_bursts
#   pseudo_candidates
#
# These were created by the prior burst/pseudo-event analysis.
# =====================================================================

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

PRE_OFFSETS = np.array(
    [-3, -2, -1],
    dtype=int,
)

POST_OFFSETS = np.array(
    [0, 1, 2],
    dtype=int,
)

ALL_OFFSETS = np.concatenate(
    [
        PRE_OFFSETS,
        POST_OFFSETS,
    ]
)

LARGE_INVALID_COST = 1e6


# ---------------------------------------------------------------------
# Validate required objects
# ---------------------------------------------------------------------

for required_object in [
    "event_source",
    "real_bursts",
    "pseudo_candidates",
]:
    if required_object not in globals():
        raise NameError(
            f"{required_object} does not exist. "
            "Run the burst-detection and candidate-generation cells first."
        )


required_source_columns = {
    "_row_id",
    "subject",
    "sequence_id",
    "trial_in_sequence",
    "js_to_next",
    "negative_feedback",
    "hard_state",
}

missing_source_columns = (
    required_source_columns
    - set(event_source.columns)
)

if missing_source_columns:
    raise KeyError(
        "event_source is missing: "
        f"{sorted(missing_source_columns)}"
    )


# ---------------------------------------------------------------------
# Validate matching-table columns
# ---------------------------------------------------------------------

required_real_burst_columns = {
    "pair_id",
    "_row_id",
    "subject",
    "sequence_id",
    "negative_feedback",
    "hard_state",
    "position_bin",
    "trial_fraction",
}

required_pseudo_candidate_columns = {
    "_row_id",
    "subject",
    "sequence_id",
    "negative_feedback",
    "hard_state",
    "position_bin",
    "trial_fraction",
}


missing_real_columns = (
    required_real_burst_columns
    - set(real_bursts.columns)
)

if missing_real_columns:
    raise KeyError(
        "real_bursts is missing: "
        f"{sorted(missing_real_columns)}"
    )


missing_pseudo_columns = (
    required_pseudo_candidate_columns
    - set(pseudo_candidates.columns)
)

if missing_pseudo_columns:
    raise KeyError(
        "pseudo_candidates is missing: "
        f"{sorted(missing_pseudo_columns)}"
    )


# Matching identifiers should be unique in their respective tables.
if not real_bursts["pair_id"].is_unique:
    raise ValueError(
        "real_bursts contains duplicated pair_id values."
    )

if not pseudo_candidates["_row_id"].is_unique:
    raise ValueError(
        "pseudo_candidates contains duplicated _row_id values."
    )


print(
    "Validation passed:",
    f"{len(real_bursts):,} real bursts and "
    f"{len(pseudo_candidates):,} pseudo-event candidates."
)


# ---------------------------------------------------------------------
# Statistical helper
# ---------------------------------------------------------------------

def signed_rank_summary(
    differences,
    comparison,
):
    differences = (
        pd.Series(
            differences,
            dtype=float,
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if len(differences) < 3:
        return {
            "comparison": comparison,
            "n": len(differences),
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "W": np.nan,
            "p": np.nan,
            "dz": np.nan,
        }

    if np.allclose(
        differences.to_numpy(),
        0,
    ):
        statistic = 0.0
        p_value = 1.0

    else:
        statistic, p_value = wilcoxon(
            differences
        )

    difference_sd = differences.std(
        ddof=1
    )

    dz = (
        differences.mean()
        / difference_sd
        if (
            np.isfinite(difference_sd)
            and difference_sd > 0
        )
        else np.nan
    )

    return {
        "comparison": comparison,
        "n": len(differences),
        "mean_difference": differences.mean(),
        "median_difference": differences.median(),
        "W": statistic,
        "p": p_value,
        "dz": dz,
    }


# ---------------------------------------------------------------------
# Main matching and analysis function
# ---------------------------------------------------------------------

def run_median_delta_robustness(
    specification_name,
    max_position_bin_difference=None,
    max_trial_fraction_difference=None,
):
    """
    Perform strict one-to-one burst/control matching and test
    subject-level median event ΔJSD.

    Parameters
    ----------
    specification_name : str
        Label for the robustness specification.

    max_position_bin_difference : int or None
        Maximum allowed difference between position bins.
        Use 0 for exact-bin matching.

    max_trial_fraction_difference : float or None
        Maximum allowed normalized sequence-position difference.
        Use 0.05 for a five-percentage-point criterion.
    """

    real = real_bursts.copy()
    controls = pseudo_candidates.copy()
    source = event_source.copy()

    real["subject"] = real["subject"].astype(str)
    controls["subject"] = controls["subject"].astype(str)
    source["subject"] = source["subject"].astype(str)

    # -------------------------------------------------------------
    # Exact matching variables
    # -------------------------------------------------------------

    exact_match_columns = [
        "subject",
        "sequence_id",
        "negative_feedback",
        "hard_state",
    ]

    if (
        "probabilityLeft" in real.columns
        and "probabilityLeft" in controls.columns
    ):
        exact_match_columns.append(
            "probabilityLeft"
        )

    # Avoid ambiguous NaN grouping keys.
    real = real.dropna(
        subset=exact_match_columns
        + [
            "position_bin",
            "trial_fraction",
        ]
    ).copy()

    controls = controls.dropna(
        subset=exact_match_columns
        + [
            "position_bin",
            "trial_fraction",
        ]
    ).copy()

    control_groups = {
        key: group.copy()
        for key, group in controls.groupby(
            exact_match_columns,
            sort=False,
        )
    }

    match_rows = []

    # -------------------------------------------------------------
    # One-to-one optimal matching
    # -------------------------------------------------------------

    for group_key, burst_group in real.groupby(
        exact_match_columns,
        sort=False,
    ):
        control_group = control_groups.get(
            group_key
        )

        if (
            control_group is None
            or control_group.empty
            or burst_group.empty
        ):
            continue

        burst_group = burst_group.copy()
        control_group = control_group.copy()

        burst_fraction = (
            burst_group["trial_fraction"]
            .to_numpy(dtype=float)
        )

        control_fraction = (
            control_group["trial_fraction"]
            .to_numpy(dtype=float)
        )

        burst_bins = (
            burst_group["position_bin"]
            .to_numpy(dtype=int)
        )

        control_bins = (
            control_group["position_bin"]
            .to_numpy(dtype=int)
        )

        fraction_difference = np.abs(
            burst_fraction[:, None]
            - control_fraction[None, :]
        )

        bin_difference = np.abs(
            burst_bins[:, None]
            - control_bins[None, :]
        )

        valid_pair = np.ones(
            fraction_difference.shape,
            dtype=bool,
        )

        if max_position_bin_difference is not None:
            valid_pair &= (
                bin_difference
                <= max_position_bin_difference
            )

        if max_trial_fraction_difference is not None:
            valid_pair &= (
                fraction_difference
                <= max_trial_fraction_difference
            )

        cost_matrix = np.where(
            valid_pair,
            fraction_difference,
            LARGE_INVALID_COST,
        )

        burst_indices, control_indices = (
            linear_sum_assignment(
                cost_matrix
            )
        )

        for burst_index, control_index in zip(
            burst_indices,
            control_indices,
        ):
            if not valid_pair[
                burst_index,
                control_index,
            ]:
                continue

            burst_row = burst_group.iloc[
                burst_index
            ]

            control_row = control_group.iloc[
                control_index
            ]

            match_rows.append(
                {
                    "pair_id": int(
                        burst_row["pair_id"]
                    ),
                    "pseudo_row_id": int(
                        control_row["_row_id"]
                    ),
                    "position_bin_difference": int(
                        abs(
                            burst_row["position_bin"]
                            - control_row["position_bin"]
                        )
                    ),
                    "trial_fraction_difference": float(
                        abs(
                            burst_row["trial_fraction"]
                            - control_row["trial_fraction"]
                        )
                    ),
                }
            )

    matches_spec = pd.DataFrame(
        match_rows
    )

    if matches_spec.empty:
        raise ValueError(
            f"No matches found for {specification_name}."
        )

    assert matches_spec["pair_id"].is_unique
    assert matches_spec["pseudo_row_id"].is_unique

    # -------------------------------------------------------------
    # Create observed-burst event table
    # -------------------------------------------------------------

    observed_events = (
        real
        .merge(
            matches_spec,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
        .rename(
            columns={
                "trial_in_sequence":
                    "event_trial_position",
            }
        )
    )

    observed_events["event_type"] = (
        "Observed pupil burst"
    )

    observed_events = observed_events[
        [
            "pair_id",
            "subject",
            "sequence_id",
            "event_trial_position",
            "negative_feedback",
            "hard_state",
            "event_type",
        ]
    ]

    # -------------------------------------------------------------
    # Create matched-control event table
    # -------------------------------------------------------------

    pseudo_lookup = (
        source[
            [
                "_row_id",
                "subject",
                "sequence_id",
                "trial_in_sequence",
                "negative_feedback",
                "hard_state",
            ]
        ]
        .rename(
            columns={
                "_row_id":
                    "pseudo_row_id",
                "trial_in_sequence":
                    "event_trial_position",
            }
        )
    )

    assert pseudo_lookup[
        "pseudo_row_id"
    ].is_unique

    control_events = (
        matches_spec
        .merge(
            pseudo_lookup,
            on="pseudo_row_id",
            how="left",
            validate="one_to_one",
        )
    )

    control_events["event_type"] = (
        "Matched non-burst"
    )

    control_events = control_events[
        [
            "pair_id",
            "subject",
            "sequence_id",
            "event_trial_position",
            "negative_feedback",
            "hard_state",
            "event_type",
        ]
    ]

    event_table_spec = pd.concat(
        [
            observed_events,
            control_events,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # Extract aligned JSD values
    # -------------------------------------------------------------

    offset_table = pd.DataFrame(
        {
            "relative_trial":
                ALL_OFFSETS
        }
    )

    event_table_spec["_merge_key"] = 1
    offset_table["_merge_key"] = 1

    aligned = (
        event_table_spec
        .merge(
            offset_table,
            on="_merge_key",
            how="inner",
        )
        .drop(
            columns="_merge_key"
        )
    )

    aligned[
        "target_trial_position"
    ] = (
        aligned[
            "event_trial_position"
        ]
        + aligned[
            "relative_trial"
        ]
    )

    js_lookup = (
        source[
            [
                "subject",
                "sequence_id",
                "trial_in_sequence",
                "js_to_next",
            ]
        ]
        .rename(
            columns={
                "trial_in_sequence":
                    "target_trial_position",
            }
        )
    )

    aligned = aligned.merge(
        js_lookup,
        on=[
            "subject",
            "sequence_id",
            "target_trial_position",
        ],
        how="left",
        validate="many_to_one",
    )

    aligned["period"] = np.where(
        aligned["relative_trial"].isin(
            PRE_OFFSETS
        ),
        "pre",
        "post",
    )

    # -------------------------------------------------------------
    # Calculate event-level ΔJSD
    # -------------------------------------------------------------

    event_period = (
        aligned
        .groupby(
            [
                "pair_id",
                "subject",
                "negative_feedback",
                "hard_state",
                "event_type",
                "period",
            ],
            as_index=False,
        )
        .agg(
            mean_js=(
                "js_to_next",
                "mean",
            ),
            n_valid=(
                "js_to_next",
                "count",
            ),
        )
    )

    event_wide = (
        event_period
        .pivot_table(
            index=[
                "pair_id",
                "subject",
                "negative_feedback",
                "hard_state",
                "event_type",
            ],
            columns="period",
            values=[
                "mean_js",
                "n_valid",
            ],
            aggfunc="first",
        )
    )

    event_wide.columns = [
        f"{measure}_{period}"
        for measure, period
        in event_wide.columns
    ]

    event_delta_spec = (
        event_wide
        .reset_index()
    )

    event_delta_spec = (
        event_delta_spec.loc[
            event_delta_spec[
                "n_valid_pre"
            ].eq(
                len(PRE_OFFSETS)
            )
            & event_delta_spec[
                "n_valid_post"
            ].eq(
                len(POST_OFFSETS)
            )
        ]
        .copy()
    )

    event_delta_spec[
        "delta_js"
    ] = (
        event_delta_spec[
            "mean_js_post"
        ]
        - event_delta_spec[
            "mean_js_pre"
        ]
    )

    event_delta_spec[
        "feedback_label"
    ] = np.where(
        event_delta_spec[
            "negative_feedback"
        ].eq(1),
        "Negative feedback",
        "Rewarded",
    )

    # -------------------------------------------------------------
    # SUBJECT-LEVEL MEDIAN EVENT ΔJSD
    # -------------------------------------------------------------

    subject_median_delta = (
        event_delta_spec
        .groupby(
            [
                "subject",
                "negative_feedback",
                "feedback_label",
                "event_type",
            ],
            as_index=False,
        )
        .agg(
            median_delta_js=(
                "delta_js",
                "median",
            ),
            n_events=(
                "pair_id",
                "nunique",
            ),
        )
    )

    population_summary = (
        subject_median_delta
        .groupby(
            [
                "feedback_label",
                "event_type",
            ],
            as_index=False,
        )
        .agg(
            mean_subject_median=(
                "median_delta_js",
                "mean",
            ),
            median_subject_median=(
                "median_delta_js",
                "median",
            ),
            sem_subject_median=(
                "median_delta_js",
                "sem",
            ),
            n_subjects=(
                "subject",
                "nunique",
            ),
        )
    )

    # -------------------------------------------------------------
    # Statistical tests
    # -------------------------------------------------------------

    results = []

    # A. Observed negative-feedback bursts versus rewarded bursts.
    observed_wide = (
        subject_median_delta.loc[
            subject_median_delta[
                "event_type"
            ]
            == "Observed pupil burst"
        ]
        .pivot_table(
            index="subject",
            columns="feedback_label",
            values="median_delta_js",
            aggfunc="first",
        )
        .dropna(
            subset=[
                "Negative feedback",
                "Rewarded",
            ]
        )
    )

    results.append(
        signed_rank_summary(
            observed_wide[
                "Negative feedback"
            ]
            - observed_wide[
                "Rewarded"
            ],
            (
                "Observed negative-feedback "
                "minus rewarded"
            ),
        )
    )

    # B. Rewarded burst versus matched rewarded control.
    rewarded_wide = (
        subject_median_delta.loc[
            subject_median_delta[
                "feedback_label"
            ]
            == "Rewarded"
        ]
        .pivot_table(
            index="subject",
            columns="event_type",
            values="median_delta_js",
            aggfunc="first",
        )
        .dropna(
            subset=[
                "Observed pupil burst",
                "Matched non-burst",
            ]
        )
    )

    results.append(
        signed_rank_summary(
            rewarded_wide[
                "Observed pupil burst"
            ]
            - rewarded_wide[
                "Matched non-burst"
            ],
            (
                "Rewarded burst minus "
                "matched non-burst"
            ),
        )
    )

    # C. Negative-feedback burst versus matched control.
    negative_wide = (
        subject_median_delta.loc[
            subject_median_delta[
                "feedback_label"
            ]
            == "Negative feedback"
        ]
        .pivot_table(
            index="subject",
            columns="event_type",
            values="median_delta_js",
            aggfunc="first",
        )
        .dropna(
            subset=[
                "Observed pupil burst",
                "Matched non-burst",
            ]
        )
    )

    results.append(
        signed_rank_summary(
            negative_wide[
                "Observed pupil burst"
            ]
            - negative_wide[
                "Matched non-burst"
            ],
            (
                "Negative-feedback burst minus "
                "matched non-burst"
            ),
        )
    )

    # D. Difference-in-differences.
    did = (
        subject_median_delta
        .pivot_table(
            index="subject",
            columns=[
                "feedback_label",
                "event_type",
            ],
            values="median_delta_js",
            aggfunc="first",
        )
    )

    required_did_columns = [
        (
            "Negative feedback",
            "Observed pupil burst",
        ),
        (
            "Negative feedback",
            "Matched non-burst",
        ),
        (
            "Rewarded",
            "Observed pupil burst",
        ),
        (
            "Rewarded",
            "Matched non-burst",
        ),
    ]

    did = did.dropna(
        subset=required_did_columns
    )

    did_difference = (
        (
            did[
                (
                    "Negative feedback",
                    "Observed pupil burst",
                )
            ]
            - did[
                (
                    "Negative feedback",
                    "Matched non-burst",
                )
            ]
        )
        - (
            did[
                (
                    "Rewarded",
                    "Observed pupil burst",
                )
            ]
            - did[
                (
                    "Rewarded",
                    "Matched non-burst",
                )
            ]
        )
    )

    results.append(
        signed_rank_summary(
            did_difference,
            "Difference-in-differences",
        )
    )

    results = pd.DataFrame(
        results
    )

    results.insert(
        0,
        "specification",
        specification_name,
    )

    diagnostics = {
        "specification":
            specification_name,
        "eligible_bursts":
            len(real),
        "matched_pairs":
            len(matches_spec),
        "retained_fraction":
            len(matches_spec) / len(real),
        "exact_bin_fraction":
            matches_spec[
                "position_bin_difference"
            ].eq(0).mean(),
        "median_position_difference":
            matches_spec[
                "trial_fraction_difference"
            ].median(),
        "maximum_position_difference":
            matches_spec[
                "trial_fraction_difference"
            ].max(),
        "complete_event_rows":
            len(event_delta_spec),
        "n_subjects":
            subject_median_delta[
                "subject"
            ].nunique(),
    }

    return {
        "matches": matches_spec,
        "event_delta": event_delta_spec,
        "subject_median_delta":
            subject_median_delta,
        "population_summary":
            population_summary,
        "results": results,
        "diagnostics": diagnostics,
    }


# =====================================================================
# RUN ROBUSTNESS SPECIFICATIONS
# =====================================================================

robustness_outputs = {}


# ---------------------------------------------------------------------
# 1. Exact position-bin matching
# ---------------------------------------------------------------------

robustness_outputs[
    "Exact position bin"
] = run_median_delta_robustness(
    specification_name=(
        "Exact position bin"
    ),
    max_position_bin_difference=0,
    max_trial_fraction_difference=None,
)


# ---------------------------------------------------------------------
# 2. Maximum 5% sequence-position difference
# ---------------------------------------------------------------------

robustness_outputs[
    "Within 5% sequence position"
] = run_median_delta_robustness(
    specification_name=(
        "Within 5% sequence position"
    ),
    max_position_bin_difference=None,
    max_trial_fraction_difference=0.05,
)


# ---------------------------------------------------------------------
# 3. Exact bin and maximum 5% difference
# ---------------------------------------------------------------------

robustness_outputs[
    "Exact bin and within 5%"
] = run_median_delta_robustness(
    specification_name=(
        "Exact bin and within 5%"
    ),
    max_position_bin_difference=0,
    max_trial_fraction_difference=0.05,
)


# =====================================================================
# COMBINE AND PRINT RESULTS
# =====================================================================

diagnostic_table = pd.DataFrame(
    [
        output["diagnostics"]
        for output
        in robustness_outputs.values()
    ]
)

result_table = pd.concat(
    [
        output["results"]
        for output
        in robustness_outputs.values()
    ],
    ignore_index=True,
)

population_table = pd.concat(
    [
        output[
            "population_summary"
        ].assign(
            specification=specification
        )
        for specification, output
        in robustness_outputs.items()
    ],
    ignore_index=True,
)


print(
    "\n"
    + "=" * 110
)

print(
    "MATCHING DIAGNOSTICS"
)

print(
    "=" * 110
)

print(
    diagnostic_table.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.6f}"
        ),
    )
)


print(
    "\n"
    + "=" * 110
)

print(
    "SUBJECT-LEVEL MEDIAN ΔJSD RESULTS"
)

print(
    "=" * 110
)

print(
    result_table.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.6f}"
        ),
    )
)


print(
    "\n"
    + "=" * 110
)

print(
    "POPULATION SUMMARY OF SUBJECT MEDIANS"
)

print(
    "=" * 110
)

print(
    population_table[
        [
            "specification",
            "feedback_label",
            "event_type",
            "mean_subject_median",
            "median_subject_median",
            "sem_subject_median",
            "n_subjects",
        ]
    ]
    .sort_values(
        [
            "specification",
            "feedback_label",
            "event_type",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.6f}"
        ),
    )
)



# ===== CELL 58 =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ============================================================
# NULL-RESULT SUMMARY FIGURE FOR STIMULUS-LOCKED PUPIL BURSTS
#
# Uses the subject-level MEDIAN ΔJSD robustness results.
#
# Default:
#   strictest matching = exact bin AND within 5%
#
# Change SPECIFICATION below if you want one of the other
# robustness variants.
# ============================================================

SPECIFICATION = "Exact bin and within 5%"
# Alternatives:
# "Exact position bin"
# "Within 5% sequence position"

if "robustness_outputs" not in globals():
    raise NameError(
        "robustness_outputs not found. "
        "Run the median robustness cell first."
    )

if SPECIFICATION not in robustness_outputs:
    raise KeyError(
        f"{SPECIFICATION!r} not found in robustness_outputs."
    )

subject_median_delta = robustness_outputs[SPECIFICATION]["subject_median_delta"].copy()
results_df = robustness_outputs[SPECIFICATION]["results"].copy()

subject_median_delta["subject"] = subject_median_delta["subject"].astype(str)

# ------------------------------------------------------------
# Helper: pull p-values / effect sizes from the saved results
# ------------------------------------------------------------

def lookup_result(label):
    row = results_df.loc[
        results_df["comparison"] == label
    ]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

res_observed_nf_minus_r = lookup_result("Observed negative-feedback minus rewarded")
res_rewarded_burst_minus_control = lookup_result("Rewarded burst minus matched non-burst")
res_nf_burst_minus_control = lookup_result("Negative-feedback burst minus matched non-burst")
res_did = lookup_result("Difference-in-differences")


def format_p(p):
    if pd.isna(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


# ------------------------------------------------------------
# Build wide tables for paired contrasts
# ------------------------------------------------------------

wide = subject_median_delta.pivot_table(
    index="subject",
    columns=["feedback_label", "event_type"],
    values="median_delta_js",
    aggfunc="first"
)

# Contrast 1: observed NF burst - observed rewarded burst
contrast1 = (
    wide[("Negative feedback", "Observed pupil burst")]
    - wide[("Rewarded", "Observed pupil burst")]
).dropna()

# Contrast 2: observed NF burst - matched NF control
contrast2 = (
    wide[("Negative feedback", "Observed pupil burst")]
    - wide[("Negative feedback", "Matched non-burst")]
).dropna()

# Contrast 3: difference-in-differences
contrast3 = (
    (
        wide[("Negative feedback", "Observed pupil burst")]
        - wide[("Negative feedback", "Matched non-burst")]
    )
    - (
        wide[("Rewarded", "Observed pupil burst")]
        - wide[("Rewarded", "Matched non-burst")]
    )
).dropna()

contrast_data = [
    ("NF burst − rewarded burst", contrast1, res_observed_nf_minus_r),
    ("NF burst − NF matched", contrast2, res_nf_burst_minus_control),
    ("Difference-in-differences", contrast3, res_did),
]

# ============================================================
# REVISED FIGURE WITH IMPROVED SPACING
# ============================================================

rng = np.random.default_rng(0)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14.5, 6.2),
    gridspec_kw={
        "width_ratios": [1.25, 1.0],
        "wspace": 0.18,
    },
)

# Leave explicit space for:
#   - the two-line figure title
#   - panel annotations
#   - multiline x-axis labels
fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.23,
    top=0.76,
    wspace=0.18,
)


# ============================================================
# PANEL A
# Four-condition subject-level medians
# ============================================================

ax = axes[0]

condition_order = [
    ("Rewarded", "Matched non-burst"),
    ("Rewarded", "Observed pupil burst"),
    ("Negative feedback", "Matched non-burst"),
    ("Negative feedback", "Observed pupil burst"),
]

# More separation between the rewarded and negative-feedback groups.
x_positions = {
    ("Rewarded", "Matched non-burst"): 0.0,
    ("Rewarded", "Observed pupil burst"): 1.0,
    ("Negative feedback", "Matched non-burst"): 3.0,
    ("Negative feedback", "Observed pupil burst"): 4.0,
}


# ------------------------------------------------------------
# Paired lines within feedback condition
# ------------------------------------------------------------

for feedback_label in [
    "Rewarded",
    "Negative feedback",
]:
    paired = (
        subject_median_delta.loc[
            subject_median_delta[
                "feedback_label"
            ]
            == feedback_label
        ]
        .pivot_table(
            index="subject",
            columns="event_type",
            values="median_delta_js",
            aggfunc="first",
        )
        .dropna(
            subset=[
                "Matched non-burst",
                "Observed pupil burst",
            ]
        )
    )

    x_control = x_positions[
        (
            feedback_label,
            "Matched non-burst",
        )
    ]

    x_burst = x_positions[
        (
            feedback_label,
            "Observed pupil burst",
        )
    ]

    for _, row in paired.iterrows():
        ax.plot(
            [
                x_control,
                x_burst,
            ],
            [
                row["Matched non-burst"],
                row["Observed pupil burst"],
            ],
            alpha=0.10,
            linewidth=0.7,
            zorder=1,
        )


# ------------------------------------------------------------
# Subject points and population summaries
# ------------------------------------------------------------

for feedback_label, event_type in condition_order:
    x_position = x_positions[
        (
            feedback_label,
            event_type,
        )
    ]

    values = (
        subject_median_delta.loc[
            (
                subject_median_delta[
                    "feedback_label"
                ]
                == feedback_label
            )
            & (
                subject_median_delta[
                    "event_type"
                ]
                == event_type
            ),
            "median_delta_js",
        ]
        .dropna()
    )

    jitter = rng.uniform(
        -0.09,
        0.09,
        size=len(values),
    )

    ax.scatter(
        np.full(
            len(values),
            x_position,
        )
        + jitter,
        values,
        alpha=0.28,
        s=18,
        zorder=2,
    )

    ax.errorbar(
        x_position,
        values.mean(),
        yerr=values.sem(),
        marker="o",
        markersize=8,
        capsize=4,
        linewidth=2,
        zorder=4,
    )


# ------------------------------------------------------------
# Panel A limits and brackets
# ------------------------------------------------------------

panel_a_values = (
    subject_median_delta[
        "median_delta_js"
    ]
    .dropna()
)

panel_a_min = panel_a_values.min()
panel_a_max = panel_a_values.max()
panel_a_range = panel_a_max - panel_a_min

if panel_a_range <= 0:
    panel_a_range = 1.0


def add_bracket(
    axis,
    x1,
    x2,
    y,
    height,
    text,
):
    axis.plot(
        [
            x1,
            x1,
            x2,
            x2,
        ],
        [
            y,
            y + height,
            y + height,
            y,
        ],
        color="black",
        linewidth=1.2,
        clip_on=False,
    )

    axis.text(
        (x1 + x2) / 2,
        y + height + 0.015 * panel_a_range,
        text,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=9,
        linespacing=1.15,
        clip_on=False,
    )


rewarded_annotation = ""

if res_rewarded_burst_minus_control is not None:
    rewarded_annotation = (
        f"{format_p(res_rewarded_burst_minus_control['p'])}\n"
        f"dz={res_rewarded_burst_minus_control['dz']:.2f}"
    )


negative_annotation = ""

if res_nf_burst_minus_control is not None:
    negative_annotation = (
        f"{format_p(res_nf_burst_minus_control['p'])}\n"
        f"dz={res_nf_burst_minus_control['dz']:.2f}"
    )


bracket_height = 0.015 * panel_a_range
rewarded_bracket_y = panel_a_max + 0.08 * panel_a_range
negative_bracket_y = panel_a_max + 0.18 * panel_a_range

add_bracket(
    ax,
    x_positions[
        (
            "Rewarded",
            "Matched non-burst",
        )
    ],
    x_positions[
        (
            "Rewarded",
            "Observed pupil burst",
        )
    ],
    rewarded_bracket_y,
    bracket_height,
    rewarded_annotation,
)

add_bracket(
    ax,
    x_positions[
        (
            "Negative feedback",
            "Matched non-burst",
        )
    ],
    x_positions[
        (
            "Negative feedback",
            "Observed pupil burst",
        )
    ],
    negative_bracket_y,
    bracket_height,
    negative_annotation,
)


ax.axhline(
    0,
    color="black",
    linewidth=0.8,
    alpha=0.55,
)

ax.set_xlim(
    -0.45,
    4.45,
)

ax.set_ylim(
    panel_a_min - 0.12 * panel_a_range,
    panel_a_max + 0.36 * panel_a_range,
)

ax.set_xticks(
    [
        0,
        1,
        3,
        4,
    ]
)

ax.set_xticklabels(
    [
        "Matched\nnon-burst",
        "Observed\npupil burst",
        "Matched\nnon-burst",
        "Observed\npupil burst",
    ],
    fontsize=9,
)

ax.tick_params(
    axis="x",
    pad=7,
)

# Higher-level feedback-condition labels beneath the tick labels.
ax.text(
    0.5,
    -0.22,
    "Rewarded trials",
    transform=ax.get_xaxis_transform(),
    horizontalalignment="center",
    verticalalignment="top",
    fontsize=10,
    fontweight="bold",
    clip_on=False,
)

ax.text(
    3.5,
    -0.22,
    "Negative-feedback trials",
    transform=ax.get_xaxis_transform(),
    horizontalalignment="center",
    verticalalignment="top",
    fontsize=10,
    fontweight="bold",
    clip_on=False,
)

ax.set_ylabel(
    "Subject-level median ΔJSD\n"
    "(mean post-event JSD − mean pre-event JSD)"
)

ax.set_title(
    "A. Stimulus-locked pupil bursts versus matched controls",
    fontsize=12,
    pad=12,
)


# ============================================================
# PANEL B
# Key within-subject contrasts
# ============================================================

ax = axes[1]

contrast_positions = [
    0.0,
    1.7,
    3.4,
]

contrast_labels = [
    "Observed NF burst\n− rewarded burst",
    "NF burst\n− NF matched",
    "Difference-in-\ndifferences",
]


# Collect limits before adding annotations.
all_contrast_values = pd.concat(
    [
        pd.Series(values).dropna()
        for _, values, _ in contrast_data
    ],
    ignore_index=True,
)

panel_b_min = all_contrast_values.min()
panel_b_max = all_contrast_values.max()
panel_b_range = panel_b_max - panel_b_min

if panel_b_range <= 0:
    panel_b_range = 1.0


for index, (
    x_position,
    (
        label,
        values,
        result,
    ),
) in enumerate(
    zip(
        contrast_positions,
        contrast_data,
    )
):
    values = (
        pd.Series(values)
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    jitter = rng.uniform(
        -0.13,
        0.13,
        size=len(values),
    )

    ax.scatter(
        np.full(
            len(values),
            x_position,
        )
        + jitter,
        values,
        alpha=0.27,
        s=18,
        zorder=2,
    )

    ax.errorbar(
        x_position,
        values.mean(),
        yerr=values.sem(),
        marker="o",
        markersize=8,
        capsize=4,
        linewidth=2,
        zorder=4,
    )

    if result is not None:
        annotation = (
            f"{format_p(result['p'])}\n"
            f"dz={result['dz']:.2f}\n"
            f"n={int(result['n'])}"
        )

        # Stagger annotations slightly to avoid horizontal crowding.
        annotation_y = (
            panel_b_max
            + (
                0.10
                + index * 0.07
            )
            * panel_b_range
        )

        ax.text(
            x_position,
            annotation_y,
            annotation,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=8.5,
            linespacing=1.15,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
                "pad": 1.5,
            },
            clip_on=False,
        )


ax.axhline(
    0,
    color="black",
    linewidth=0.8,
    alpha=0.55,
)

ax.set_xlim(
    -0.5,
    3.9,
)

ax.set_ylim(
    panel_b_min - 0.12 * panel_b_range,
    panel_b_max + 0.42 * panel_b_range,
)

ax.set_xticks(
    contrast_positions,
)

ax.set_xticklabels(
    contrast_labels,
    fontsize=9,
)

ax.tick_params(
    axis="x",
    pad=7,
)

ax.set_ylabel(
    "Within-subject contrast in median ΔJSD"
)

ax.set_title(
    "B. Key hypothesis contrasts",
    fontsize=12,
    pad=12,
)


# ============================================================
# Figure-level title
# ============================================================

fig.suptitle(
    "Stimulus-locked pupil bursts do not robustly add state lability "
    "beyond negative feedback itself\n"
    f"Robustness specification: {SPECIFICATION}",
    x=0.53,
    y=0.985,
    fontsize=13,
    linespacing=1.25,
)


plt.savefig(
    "output/stimulus_locked_burst_null_summary_spaced.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()



# ===== CELL 59 =====
# =====================================================================
# CREATE SLIDE-READY FIGURE GROUPS
#
# Outputs:
#   1. slide_1_tonic_state_physiology.png
#   2. slide_2_feedback_state_lability.png
#   3. slide_3_phasic_burst_null_result.png
#
# Canvas size: 1920 × 1080 pixels, suitable for 16:9 slides.
# =====================================================================

from pathlib import Path

import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageOps,
)
from matplotlib import font_manager


# =====================================================================
# 1. FILE CONFIGURATION
# =====================================================================

OUTPUT_DIRECTORY = Path("output")
SLIDE_DIRECTORY = OUTPUT_DIRECTORY / "slide_ready"
SLIDE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def first_existing_path(candidates):
    """
    Return the first existing path from a list of candidate filenames.
    """
    for candidate in candidates:
        candidate = Path(candidate)

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "None of these candidate files were found:\n"
        + "\n".join(
            str(candidate)
            for candidate in candidates
        )
    )


# ---------------------------------------------------------------------
# Slide 1: tonic pupil and latent state
# ---------------------------------------------------------------------

RAW_TONIC_FIGURE = first_existing_path(
    [
        OUTPUT_DIRECTORY
        / "tonic_pupil_by_epoch_and_pooled_violin.png",

        OUTPUT_DIRECTORY
        / "tonic_pupil_raw_by_epoch_and_pooled_violin.png",

        OUTPUT_DIRECTORY
        / "tonic_pupil_raw_by_epoch_and_state.png",
    ]
)

ZSCORED_TONIC_FIGURE = first_existing_path(
    [
        OUTPUT_DIRECTORY
        / "tonic_pupil_zscored_by_epoch_and_pooled_violin.png",

        OUTPUT_DIRECTORY
        / "tonic_pupil_zscored_by_epoch_and_state.png",
    ]
)


# ---------------------------------------------------------------------
# Slide 2: negative feedback and posterior lability
# ---------------------------------------------------------------------

FEEDBACK_LABILITY_FIGURE = first_existing_path(
    [
        OUTPUT_DIRECTORY
        / "jsd_burst_vs_matched_pseudoevents.png",

        OUTPUT_DIRECTORY
        / "jsd_timelocked_to_phasic_pupil_bursts.png",
    ]
)


# ---------------------------------------------------------------------
# Slide 3: phasic-burst null result
# ---------------------------------------------------------------------

PHASIC_NULL_FIGURE = first_existing_path(
    [
        OUTPUT_DIRECTORY
        / "stimulus_locked_burst_null_summary_spaced.png",

        OUTPUT_DIRECTORY
        / "stimulus_locked_burst_null_summary.png",
    ]
)


print("Using figures:")
print("  Raw tonic:", RAW_TONIC_FIGURE)
print("  Z-scored tonic:", ZSCORED_TONIC_FIGURE)
print("  Feedback lability:", FEEDBACK_LABILITY_FIGURE)
print("  Phasic null:", PHASIC_NULL_FIGURE)


# =====================================================================
# 2. SLIDE STYLE
# =====================================================================

SLIDE_WIDTH = 1920
SLIDE_HEIGHT = 1080

BACKGROUND = "white"
TEXT_COLOR = (25, 25, 25)
MUTED_TEXT_COLOR = (75, 75, 75)
BORDER_COLOR = (205, 205, 205)
CARD_BACKGROUND = (247, 247, 247)
ACCENT_BACKGROUND = (235, 240, 245)

LEFT_MARGIN = 85
RIGHT_MARGIN = 85
TOP_MARGIN = 55
BOTTOM_MARGIN = 55

TITLE_HEIGHT = 112
TAKEAWAY_HEIGHT = 105

PANEL_GAP = 36


# ---------------------------------------------------------------------
# Cross-platform fonts
# ---------------------------------------------------------------------

regular_font_path = font_manager.findfont(
    "DejaVu Sans"
)

bold_font_path = font_manager.findfont(
    font_manager.FontProperties(
        family="DejaVu Sans",
        weight="bold",
    )
)

TITLE_FONT = ImageFont.truetype(
    bold_font_path,
    47,
)

SUBTITLE_FONT = ImageFont.truetype(
    regular_font_path,
    25,
)

PANEL_LABEL_FONT = ImageFont.truetype(
    bold_font_path,
    28,
)

CARD_HEADER_FONT = ImageFont.truetype(
    bold_font_path,
    28,
)

CARD_TEXT_FONT = ImageFont.truetype(
    regular_font_path,
    24,
)

TAKEAWAY_FONT = ImageFont.truetype(
    bold_font_path,
    27,
)

FOOTNOTE_FONT = ImageFont.truetype(
    regular_font_path,
    19,
)


# =====================================================================
# 3. IMAGE HELPERS
# =====================================================================

def trim_white_border(
    image,
    threshold=248,
    padding=10,
):
    """
    Remove nearly white empty margins from a figure.
    """
    image = image.convert("RGB")
    array = np.asarray(image)

    nonwhite = np.any(
        array < threshold,
        axis=2,
    )

    if not nonwhite.any():
        return image

    rows, columns = np.where(
        nonwhite
    )

    left = max(
        int(columns.min()) - padding,
        0,
    )

    upper = max(
        int(rows.min()) - padding,
        0,
    )

    right = min(
        int(columns.max()) + padding + 1,
        image.width,
    )

    lower = min(
        int(rows.max()) + padding + 1,
        image.height,
    )

    return image.crop(
        (
            left,
            upper,
            right,
            lower,
        )
    )


def fractional_crop(
    image,
    crop_fraction=None,
):
    """
    Crop using fractional coordinates:
        (left, upper, right, lower)

    Example:
        (0.00, 0.08, 1.00, 0.98)
    """
    if crop_fraction is None:
        return image

    left_fraction, upper_fraction, right_fraction, lower_fraction = (
        crop_fraction
    )

    return image.crop(
        (
            int(
                left_fraction
                * image.width
            ),
            int(
                upper_fraction
                * image.height
            ),
            int(
                right_fraction
                * image.width
            ),
            int(
                lower_fraction
                * image.height
            ),
        )
    )


def fit_image_to_box(
    image,
    width,
    height,
):
    """
    Resize an image to fit within a box while preserving aspect ratio.
    """
    image = image.copy()

    image.thumbnail(
        (
            int(width),
            int(height),
        ),
        Image.Resampling.LANCZOS,
    )

    return image


def draw_centered_text(
    draw,
    text,
    box,
    font,
    fill=TEXT_COLOR,
    spacing=6,
):
    """
    Draw multiline text centered within a bounding box.
    """
    x1, y1, x2, y2 = box

    text_box = draw.multiline_textbbox(
        (
            0,
            0,
        ),
        text,
        font=font,
        spacing=spacing,
        align="center",
    )

    text_width = (
        text_box[2]
        - text_box[0]
    )

    text_height = (
        text_box[3]
        - text_box[1]
    )

    x = (
        x1
        + (
            x2 - x1
            - text_width
        ) / 2
    )

    y = (
        y1
        + (
            y2 - y1
            - text_height
        ) / 2
    )

    draw.multiline_text(
        (
            x,
            y,
        ),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def paste_figure_panel(
    canvas,
    draw,
    image_path,
    box,
    panel_label=None,
    crop_fraction=None,
):
    """
    Insert one existing figure inside a bordered slide panel.
    """
    x1, y1, x2, y2 = box

    draw.rounded_rectangle(
        box,
        radius=18,
        fill="white",
        outline=BORDER_COLOR,
        width=2,
    )

    label_height = (
        44
        if panel_label
        else 16
    )

    if panel_label:
        draw.text(
            (
                x1 + 18,
                y1 + 10,
            ),
            panel_label,
            font=PANEL_LABEL_FONT,
            fill=TEXT_COLOR,
        )

    figure = Image.open(
        image_path
    ).convert("RGB")

    figure = fractional_crop(
        figure,
        crop_fraction=crop_fraction,
    )

    figure = trim_white_border(
        figure
    )

    available_width = (
        x2 - x1 - 28
    )

    available_height = (
        y2
        - y1
        - label_height
        - 22
    )

    figure = fit_image_to_box(
        figure,
        available_width,
        available_height,
    )

    paste_x = int(
        x1
        + (
            x2 - x1
            - figure.width
        ) / 2
    )

    paste_y = int(
        y1
        + label_height
        + (
            available_height
            - figure.height
        ) / 2
    )

    canvas.paste(
        figure,
        (
            paste_x,
            paste_y,
        ),
    )


def draw_title(
    draw,
    title,
    subtitle=None,
):
    draw.text(
        (
            LEFT_MARGIN,
            TOP_MARGIN,
        ),
        title,
        font=TITLE_FONT,
        fill=TEXT_COLOR,
    )

    if subtitle:
        draw.text(
            (
                LEFT_MARGIN,
                TOP_MARGIN + 58,
            ),
            subtitle,
            font=SUBTITLE_FONT,
            fill=MUTED_TEXT_COLOR,
        )


def draw_takeaway(
    draw,
    text,
):
    y1 = (
        SLIDE_HEIGHT
        - BOTTOM_MARGIN
        - TAKEAWAY_HEIGHT
    )

    y2 = (
        SLIDE_HEIGHT
        - BOTTOM_MARGIN
    )

    draw.rounded_rectangle(
        (
            LEFT_MARGIN,
            y1,
            SLIDE_WIDTH - RIGHT_MARGIN,
            y2,
        ),
        radius=18,
        fill=ACCENT_BACKGROUND,
    )

    draw_centered_text(
        draw,
        text,
        (
            LEFT_MARGIN + 24,
            y1 + 8,
            SLIDE_WIDTH
            - RIGHT_MARGIN
            - 24,
            y2 - 8,
        ),
        font=TAKEAWAY_FONT,
    )


def save_slide(
    canvas,
    filename,
):
    output_path = (
        SLIDE_DIRECTORY
        / filename
    )

    canvas.save(
        output_path,
        quality=95,
    )

    print(
        "Saved:",
        output_path,
    )

    return output_path


# =====================================================================
# 4. SLIDE 1 — TONIC STATE PHYSIOLOGY
# =====================================================================

canvas = Image.new(
    "RGB",
    (
        SLIDE_WIDTH,
        SLIDE_HEIGHT,
    ),
    BACKGROUND,
)

draw = ImageDraw.Draw(
    canvas
)

draw_title(
    draw,
    "Tonic pupil covaries with GLM-HMM behavioral state",
    (
        "Raw pupil measurements and within-animal "
        "standardized analyses"
    ),
)

body_top = (
    TOP_MARGIN
    + TITLE_HEIGHT
    + 18
)

body_bottom = (
    SLIDE_HEIGHT
    - BOTTOM_MARGIN
    - TAKEAWAY_HEIGHT
    - 25
)

available_width = (
    SLIDE_WIDTH
    - LEFT_MARGIN
    - RIGHT_MARGIN
    - PANEL_GAP
)

panel_width = (
    available_width // 2
)

left_panel = (
    LEFT_MARGIN,
    body_top,
    LEFT_MARGIN + panel_width,
    body_bottom,
)

right_panel = (
    LEFT_MARGIN
    + panel_width
    + PANEL_GAP,
    body_top,
    SLIDE_WIDTH - RIGHT_MARGIN,
    body_bottom,
)

paste_figure_panel(
    canvas,
    draw,
    RAW_TONIC_FIGURE,
    left_panel,
    panel_label="A  Raw tonic pupil",
    # Increase the second value slightly to remove more title whitespace.
    crop_fraction=(
        0.00,
        0.03,
        1.00,
        0.99,
    ),
)

paste_figure_panel(
    canvas,
    draw,
    ZSCORED_TONIC_FIGURE,
    right_panel,
    panel_label="B  Within-animal z-score",
    crop_fraction=(
        0.00,
        0.03,
        1.00,
        0.99,
    ),
)

draw_takeaway(
    draw,
    (
        "Tonic pupil diameter differed between engaged and "
        "pooled biased states, with the clearest effect later "
        "in biased-prior blocks."
    ),
)

slide_1_path = save_slide(
    canvas,
    "slide_1_tonic_state_physiology.png",
)


# =====================================================================
# 5. SLIDE 2 — FEEDBACK AND STATE LABILITY
# =====================================================================

canvas = Image.new(
    "RGB",
    (
        SLIDE_WIDTH,
        SLIDE_HEIGHT,
    ),
    BACKGROUND,
)

draw = ImageDraw.Draw(
    canvas
)

draw_title(
    draw,
    "Negative feedback is associated with posterior state lability",
    (
        "Failure strongly predicts subsequent state-belief movement "
        "and biased-to-engaged transitions"
    ),
)

body_top = (
    TOP_MARGIN
    + TITLE_HEIGHT
    + 18
)

body_bottom = (
    SLIDE_HEIGHT
    - BOTTOM_MARGIN
    - TAKEAWAY_HEIGHT
    - 25
)

main_figure_width = 1260
stats_card_width = (
    SLIDE_WIDTH
    - LEFT_MARGIN
    - RIGHT_MARGIN
    - PANEL_GAP
    - main_figure_width
)

figure_panel = (
    LEFT_MARGIN,
    body_top,
    LEFT_MARGIN + main_figure_width,
    body_bottom,
)

stats_panel = (
    LEFT_MARGIN
    + main_figure_width
    + PANEL_GAP,
    body_top,
    SLIDE_WIDTH - RIGHT_MARGIN,
    body_bottom,
)

paste_figure_panel(
    canvas,
    draw,
    FEEDBACK_LABILITY_FIGURE,
    figure_panel,
    panel_label="A  Event-aligned posterior movement",
    crop_fraction=(
        0.00,
        0.02,
        1.00,
        0.99,
    ),
)


# ---------------------------------------------------------------------
# Result card
# ---------------------------------------------------------------------

draw.rounded_rectangle(
    stats_panel,
    radius=18,
    fill=CARD_BACKGROUND,
    outline=BORDER_COLOR,
    width=2,
)

stats_x1, stats_y1, stats_x2, stats_y2 = (
    stats_panel
)

draw.text(
    (
        stats_x1 + 28,
        stats_y1 + 28,
    ),
    "Key effects",
    font=CARD_HEADER_FONT,
    fill=TEXT_COLOR,
)

card_lines = [
    (
        "Any inferred state switch\n"
        "within 3 trials"
    ),
    (
        "Negative feedback:\n"
        "OR = 1.90"
    ),
    (
        "Biased → engaged\n"
        "within 3 trials"
    ),
    (
        "Negative feedback:\n"
        "OR = 4.23"
    ),
]

card_y = (
    stats_y1 + 94
)

for line_index, line in enumerate(
    card_lines
):
    if line_index in {
        0,
        2,
    }:
        font = CARD_HEADER_FONT
        fill = TEXT_COLOR
    else:
        font = CARD_TEXT_FONT
        fill = MUTED_TEXT_COLOR

    draw.multiline_text(
        (
            stats_x1 + 28,
            card_y,
        ),
        line,
        font=font,
        fill=fill,
        spacing=5,
    )

    text_box = draw.multiline_textbbox(
        (
            stats_x1 + 28,
            card_y,
        ),
        line,
        font=font,
        spacing=5,
    )

    card_y = (
        text_box[3]
        + 26
    )

    if line_index == 1:
        draw.line(
            (
                stats_x1 + 28,
                card_y,
                stats_x2 - 28,
                card_y,
            ),
            fill=BORDER_COLOR,
            width=2,
        )

        card_y += 28


draw.multiline_text(
    (
        stats_x1 + 28,
        stats_y2 - 125,
    ),
    (
        "Interpretation\n"
        "Failure provides evidence that the current "
        "strategy may no longer be effective."
    ),
    font=FOOTNOTE_FONT,
    fill=MUTED_TEXT_COLOR,
    spacing=6,
)

draw_takeaway(
    draw,
    (
        "Negative feedback was the dominant predictor of state "
        "instability and strongly increased the probability of "
        "returning from a biased state to engagement."
    ),
)

slide_2_path = save_slide(
    canvas,
    "slide_2_feedback_state_lability.png",
)


# =====================================================================
# 6. SLIDE 3 — PHASIC BURST NULL RESULT
# =====================================================================

canvas = Image.new(
    "RGB",
    (
        SLIDE_WIDTH,
        SLIDE_HEIGHT,
    ),
    BACKGROUND,
)

draw = ImageDraw.Draw(
    canvas
)

draw_title(
    draw,
    (
        "Stimulus-locked phasic bursts do not robustly predict "
        "additional state lability"
    ),
    (
        "Strict one-to-one matching and subject-level median "
        "sensitivity analyses"
    ),
)

body_top = (
    TOP_MARGIN
    + TITLE_HEIGHT
    + 18
)

body_bottom = (
    SLIDE_HEIGHT
    - BOTTOM_MARGIN
    - TAKEAWAY_HEIGHT
    - 25
)

full_panel = (
    LEFT_MARGIN,
    body_top,
    SLIDE_WIDTH - RIGHT_MARGIN,
    body_bottom,
)

paste_figure_panel(
    canvas,
    draw,
    PHASIC_NULL_FIGURE,
    full_panel,
    panel_label=None,
    crop_fraction=(
        0.00,
        0.08,
        1.00,
        0.99,
    ),
)

draw_takeaway(
    draw,
    (
        "Within this dataset, phasic pupil bursts cannot be "
        "reliably said to precede increased state lability; "
        "the task was not designed to isolate controlled reversals."
    ),
)

slide_3_path = save_slide(
    canvas,
    "slide_3_phasic_burst_null_result.png",
)


# =====================================================================
# 7. FINAL SUMMARY
# =====================================================================

print(
    "\nSlide-ready assets created:"
)

for path in [
    slide_1_path,
    slide_2_path,
    slide_3_path,
]:
    print(
        " ",
        path,
    )



# ===== CELL 60 =====
from pathlib import Path

slide_dir = Path("output") / "slide_ready"

print("Slide folder:")
print(slide_dir.resolve())

print("\nFiles:")
for path in slide_dir.glob("*.png"):
    print(path.resolve())



# ===== CELL 61 =====
# =====================================================================
# EXPORT INDIVIDUAL FIGURES FOR A 16:9 PRESENTATION AT 300 DPI
#
# Outputs:
#   output/figures_300dpi/
#
# IMPORTANT:
# Re-exporting an existing raster image at 300 DPI does not create new
# scientific detail. This cell standardizes pixel dimensions, whitespace,
# and DPI metadata. Re-rendering from matplotlib at 300 DPI is preferable
# when the original plotting objects or plotting cells are available.
# =====================================================================

from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SOURCE_DIRECTORY = Path("output")

EXPORT_DIRECTORY = (
    SOURCE_DIRECTORY
    / "figures_300dpi"
)

EXPORT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Target sizes
#
# A standard widescreen PowerPoint slide is approximately:
#     13.333 × 7.5 inches
#
# These dimensions leave room for a slide title and margins.
# ---------------------------------------------------------------------

DPI = 300

# Two figures placed side by side.
HALF_SLIDE_SIZE_INCHES = (
    6.1,
    4.8,
)

# One figure occupying most of the slide width.
FULL_SLIDE_SIZE_INCHES = (
    12.3,
    5.4,
)


def inches_to_pixels(
    size_inches,
    dpi=DPI,
):
    width_inches, height_inches = (
        size_inches
    )

    return (
        int(
            round(
                width_inches * dpi
            )
        ),
        int(
            round(
                height_inches * dpi
            )
        ),
    )


HALF_SLIDE_PIXELS = inches_to_pixels(
    HALF_SLIDE_SIZE_INCHES
)

FULL_SLIDE_PIXELS = inches_to_pixels(
    FULL_SLIDE_SIZE_INCHES
)


print(
    "Half-slide dimensions:",
    HALF_SLIDE_PIXELS,
)

print(
    "Full-slide dimensions:",
    FULL_SLIDE_PIXELS,
)


# ---------------------------------------------------------------------
# Source figures and intended presentation sizes
# ---------------------------------------------------------------------

figure_configuration = [
    {
        "source": (
            SOURCE_DIRECTORY
            / "tonic_pupil_by_epoch_and_pooled_violin.png"
        ),
        "output": (
            "tonic_pupil_raw_half_slide_300dpi.png"
        ),
        "size_pixels": HALF_SLIDE_PIXELS,
    },
    {
        "source": (
            SOURCE_DIRECTORY
            / "tonic_pupil_zscored_by_epoch_and_pooled_violin.png"
        ),
        "output": (
            "tonic_pupil_zscored_half_slide_300dpi.png"
        ),
        "size_pixels": HALF_SLIDE_PIXELS,
    },
    {
        "source": (
            SOURCE_DIRECTORY
            / "jsd_burst_vs_matched_pseudoevents.png"
        ),
        "output": (
            "feedback_state_lability_full_slide_300dpi.png"
        ),
        "size_pixels": FULL_SLIDE_PIXELS,
    },
    {
        "source": (
            SOURCE_DIRECTORY
            / "stimulus_locked_burst_null_summary_spaced.png"
        ),
        "output": (
            "phasic_burst_null_full_slide_300dpi.png"
        ),
        "size_pixels": FULL_SLIDE_PIXELS,
    },
]


# ---------------------------------------------------------------------
# Remove unused white space around an existing figure
# ---------------------------------------------------------------------

def trim_near_white_border(
    image,
    threshold=248,
    padding=25,
):
    """
    Crop nearly white outer margins while retaining a small padding.
    """
    image = image.convert("RGB")

    array = np.asarray(
        image
    )

    nonwhite = np.any(
        array < threshold,
        axis=2,
    )

    if not nonwhite.any():
        return image

    rows, columns = np.where(
        nonwhite
    )

    left = max(
        int(columns.min()) - padding,
        0,
    )

    upper = max(
        int(rows.min()) - padding,
        0,
    )

    right = min(
        int(columns.max()) + padding + 1,
        image.width,
    )

    lower = min(
        int(rows.max()) + padding + 1,
        image.height,
    )

    return image.crop(
        (
            left,
            upper,
            right,
            lower,
        )
    )


# ---------------------------------------------------------------------
# Fit image onto an exact-size white canvas
# ---------------------------------------------------------------------

def export_figure(
    source_path,
    output_path,
    target_size_pixels,
    dpi=DPI,
    trim=True,
    margin_fraction=0.025,
):
    source_path = Path(
        source_path
    )

    output_path = Path(
        output_path
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Figure not found: {source_path.resolve()}"
        )

    image = Image.open(
        source_path
    ).convert("RGB")

    original_size = image.size

    if trim:
        image = trim_near_white_border(
            image
        )

    target_width, target_height = (
        target_size_pixels
    )

    horizontal_margin = int(
        target_width
        * margin_fraction
    )

    vertical_margin = int(
        target_height
        * margin_fraction
    )

    usable_width = (
        target_width
        - 2 * horizontal_margin
    )

    usable_height = (
        target_height
        - 2 * vertical_margin
    )

    scale = min(
        usable_width / image.width,
        usable_height / image.height,
    )

    resized_width = max(
        1,
        int(
            round(
                image.width * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                image.height * scale
            )
        ),
    )

    image = image.resize(
        (
            resized_width,
            resized_height,
        ),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (
            target_width,
            target_height,
        ),
        "white",
    )

    paste_x = (
        target_width
        - resized_width
    ) // 2

    paste_y = (
        target_height
        - resized_height
    ) // 2

    canvas.paste(
        image,
        (
            paste_x,
            paste_y,
        ),
    )

    canvas.save(
        output_path,
        format="PNG",
        dpi=(
            dpi,
            dpi,
        ),
        optimize=True,
    )

    print(
        f"\nSaved: {output_path.resolve()}"
    )

    print(
        f"  Source pixels: {original_size[0]} × {original_size[1]}"
    )

    print(
        f"  Output pixels: {target_width} × {target_height}"
    )

    print(
        f"  Physical size: "
        f"{target_width / dpi:.2f} × "
        f"{target_height / dpi:.2f} inches"
    )

    print(
        f"  DPI metadata: {dpi}"
    )


# ---------------------------------------------------------------------
# Export all configured figures
# ---------------------------------------------------------------------

for configuration in figure_configuration:
    export_figure(
        source_path=configuration[
            "source"
        ],
        output_path=(
            EXPORT_DIRECTORY
            / configuration[
                "output"
            ]
        ),
        target_size_pixels=configuration[
            "size_pixels"
        ],
        dpi=DPI,
    )


print(
    "\nAll 300-DPI figures were saved to:"
)

print(
    EXPORT_DIRECTORY.resolve()
)

