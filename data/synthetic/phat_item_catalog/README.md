# p-hat Item Catalog

This folder contains the copied item catalog used by the p-hat-driven synthetic
generation scripts.

The local source used for this copy was the p-hat item catalog from the working
project tree.

Copied contents:

| Family | Files | Item rows |
|---|---:|---:|
| `head_single/` | 10 | 1,000 |
| `head_combo/` | 45 | 1,035 |
| Total | 55 | 2,035 |

Each `items.txt` file uses:

```text
item_id<TAB>alpha<TAB>topic_vector
```

All copied `alpha` and `topic_vector` values are 10-dimensional, and each
`topic_vector` sums to 1 up to floating-point precision.
