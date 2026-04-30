"""Path and filename conventions for public benchmark artifacts."""

from __future__ import annotations

from pathlib import Path


SPLIT_PREFIXES = ("train__", "test__", "train_", "test_")


def public_user_filename(filename: str) -> str:
    """Return a split-neutral public filename for a per-user JSON file."""

    cleaned = Path(filename).name
    for prefix in SPLIT_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned


def phat_user_path(
    root: Path | str,
    setting: str,
    model_name: str,
    scenario_folder: str,
    filename: str,
) -> Path:
    """Return the public p-hat-driven per-user path.

    Public layout intentionally omits the legacy `all/` layer:

    data/synthetic/phat_driven_mixed/KF-AF/PS/user.json

    The split is stored in file metadata, not in the public filename.
    """

    return (
        Path(root)
        / setting
        / model_name
        / scenario_folder
        / public_user_filename(filename)
    )
