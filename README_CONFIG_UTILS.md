# Config and utility bootstrap

This folder contains the first implementation pass for the NMA pipeline refactor:

- `config.py` — project paths, canonical columns, encoding conventions, QC settings, pupil settings, GLM-HMM settings, RL baseline settings, transition-analysis settings, and figure settings.
- `utils.py` — reusable validation, encoding, sequence-safe shifting, scaling, posterior, statistical-output, and pupil-trace primitives.
- `tests/test_utils.py` — focused tests for normal, missing, constant, and malformed inputs.

## Local smoke checks

From this folder:

```bash
python -m py_compile config.py utils.py
python config.py
pytest -q
```

`config.py` creates directories only when run as a script or when `ensure_project_directories()` is called explicitly. Importing it has no filesystem, network, or authentication side effects.

The project plan is intentionally not marked complete yet. Completion should be recorded only after these files run successfully in the target environment and the user confirms the stage.
