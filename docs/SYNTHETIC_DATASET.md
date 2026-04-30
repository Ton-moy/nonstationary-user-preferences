# Synthetic Benchmark Organization

This document is the working contract for migrating the synthetic benchmark into the final anonymous repository.

## Paper-Facing Names

Use the names from the paper everywhere in public files:

| Paper name | Meaning | Public folder |
|---|---|---|
| `PS` | Preference Shifting | `PS/` |
| `PSC1T` | Preference Shifting and Conservation, one new topic | `PSC1T/` |
| `PSC2T` | Preference Shifting and Conservation, two-topic combination events | `PSC2T/` |
| `PB` | Preference Broadening | `PB/` |
| `PBC1T` | Preference Broadening and Conservation, one new topic | `PBC1T/` |
| `PBC2T` | Preference Broadening and Conservation, two-topic combination events | `PBC2T/` |

Avoid development-era labels in public paths. The final repository should expose only the paper taxonomy above and the recommendation setting names (`theta_driven`, `p_driven`, `phat_driven_top2`, `phat_driven_mixed`).

## Public Directory Layout

```text
data/synthetic/
├── README.md
├── theta_driven/
├── p_driven/
├── phat_driven_top2/
└── phat_driven_mixed/
```

For `phat_driven_top2/` and `phat_driven_mixed/`, do not keep the legacy `all/` layer under each model. The public structure should be:

```text
data/synthetic/phat_driven_mixed/
├── KF-AF/
│   ├── ps_theta/
│   ├── ps_p/
│   ├── psc1t_theta/
│   ├── psc1t_p/
│   ├── psc2t_theta/
│   ├── psc2t_p/
│   ├── pb_theta/
│   ├── pb_p/
│   ├── pbc1t_theta/
│   ├── pbc1t_p/
│   ├── pbc2t_theta/
│   └── pbc2t_p/
├── AROW/
├── BLR/
├── BLR-VB/
├── BLR-FF/
├── BLR-SW/
├── BLR-PP/
└── BLR-NIG/
```

If migrating from legacy paths such as:

```text
BLR/all/pb_p/train_pb_U1.json
```

write them as:

```text
BLR/pb_p/pb_U1.json
```

The split must live in JSON metadata (`"split": "train"` or `"split": "test"`), not in the filename.

Each generated file should contain machine-readable metadata:

```json
{
  "benchmark": "synthetic",
  "setting": "theta_driven",
  "scenario": "PS",
  "user_id": "theta_driven_PS_U0001",
  "split": "train",
  "timesteps": []
}
```

Each timestep should include:

```json
{
  "topic_vector": [0.0],
  "rating": 0.0,
  "preference_vector": [0.0],
  "preference_change_label": 1
}
```

For `phat_driven_*`, also include:

```json
{
  "generated_topic_vector": [0.0],
  "rating": 0.0,
  "preference_vector": [0.0],
  "preference_change_label": 1
}
```

## Migration Rule

Move behavior into library code:

| Legacy file | New home |
|---|---|
| `dataset/generate_self_directed_users*.py` | `src/nspb/scenarios.py`, `src/nspb/catalog.py`, `scripts/01_generate_synthetic.py` |
| `dataset/data_split.py` | `src/nspb/utils.py` |
| `dataset/generate_image*.py` | `scripts/05_make_tables_and_figures.py` |
| `dataset/models.py` | `src/nspb/models.py` |

For p-hat-driven generated users, flatten legacy source paths as follows:

| Legacy path segment | Public path segment |
|---|---|
| `<model>/all/<scenario>/<split>__<user>.json` | `<model>/<scenario>/<user>.json` |
| `train_pb_U1.json` | `pb_U1.json` |
| `test_pbc1t_U11.json` | `pbc1t_U11.json` |

The public script should be a thin CLI. It should not contain model update rules, scenario logic, or hard-coded absolute paths.

## Hyperparameter Configuration

Tuned model hyperparameters live in:

```text
configs/model_hyperparameters.json
```

The file has two groups:

| Group | Used by |
|---|---|
| `evaluation` | `evaluate_synthetic_main.py`, `evaluate_synthetic.py`, `evaluate_synthetic_distances.py`, `evaluate_phat.py` |
| `phat_generation` | `generate_phat.py` |

Evaluation scripts accept `--params-by-model-json` so a reviewer can reproduce the paper defaults or swap in another JSON file without editing Python code.

## Quality Gates

Before publishing the synthetic artifact:

- `make smoke` passes from a fresh clone.
- `make synthetic-stats` reproduces the reported Table 1 totals.
- No `.out`, `.err`, SLURM files, `__pycache__`, or local absolute paths exist.
- `tests/test_scenarios.py` checks topic vectors sum to 1 and change labels match `||p_t - p_{t-1}||_2 > 0.25`.
- `tests/test_catalog.py` or equivalent checks the p-hat catalog has exactly 2,960 items.
