# Real Dataset — Goodreads

Per-user JSON files are git-ignored. Place data under the following layout before running
the tuning or evaluation scripts:

```
data/real/
  book_topics.npz          # NPZ with arrays: book_ids (N,), topics (N, K)
  cos090/
    train/                 # per-user JSON files, cosine similarity threshold ≥ 0.90
    test/
  cos095/
    train/                 # per-user JSON files, cosine similarity threshold ≥ 0.95
    test/
```

## User JSON schema

Each file is a single JSON object:

```json
{
  "user_id": "<hash>",
  "reviews": [
    {
      "book_id": "<id>",
      "rating": 1-5,
      "timestamp": <unix seconds>,
      "timestep": <int, 1-based chronological order>,
      "pair_k_timesteps": [<timestep of paired review>, ...],
      "pair_k_details": [
        {
          "k_timestep": <int>,
          "cosine_sim_with_j": <float>,
          "rating_diff_abs": <float>
        }
      ],
      "pair_k_cosine_sim": [<float>, ...]
    }
  ]
}
```

## Quickstart

```bash
# Tune one model (MPI, 30 000 trials)
mpirun -n 16 python scripts/tune_real.py \
    --train-dir data/real/cos090/train \
    --test-dir  data/real/cos090/test \
    --book-topics-npz data/real/book_topics.npz \
    --model KF-AF --n-trials 30000 --min-rating-diff 2.0 \
    --output-dir results/real/tune/KF-AF

# Evaluate with tuned hyperparameters
mpirun -n 16 python scripts/evaluate_real.py \
    --train-dir data/real/cos090/train \
    --test-dir  data/real/cos090/test \
    --book-topics-npz data/real/book_topics.npz \
    --params-by-model-json configs/model_hyperparameters_real.json \
    --output-dir results/real/eval
```
