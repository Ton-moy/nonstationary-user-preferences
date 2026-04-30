#!/usr/bin/env python3
"""Synthetic benchmark CLI.

This script is intentionally thin. Generation logic should live in `src/nspb`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from nspb.catalog import expected_catalog_size, generate_catalog
from nspb.paths import phat_user_path
from nspb.scenarios import SCENARIOS, write_theta_p_dataset


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def print_reported_stats(config: dict) -> None:
    print("Reported synthetic benchmark totals")
    for name, setting in config["settings"].items():
        print(
            f"- {name}: users={setting['users']}, items={setting['reported_items']}, "
            f"interactions={setting['reported_interactions']}, "
            f"p_changes={setting['reported_preference_change_events']}"
        )
    totals = config["reported_totals"]
    print(
        f"- total: users={totals['users']}, items={totals['items']}, "
        f"interactions={totals['interactions']}, "
        f"p_changes={totals['preference_change_events']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synthetic.yaml", type=Path)
    parser.add_argument("--all", action="store_true", help="Generate theta/p-driven base logs.")
    parser.add_argument("--catalog", action="store_true", help="Generate the p-hat item catalog (2,960 items).")
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=["theta_driven", "p_driven"],
        default=["theta_driven", "p_driven"],
        help="Base settings to generate. p-hat generation remains in scripts/generate_phat.py.",
    )
    parser.add_argument(
        "--output-root",
        default=Path("data/synthetic"),
        type=Path,
        help="Directory where generated synthetic settings are written.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print the plan.")
    parser.add_argument("--stats-only", action="store_true", help="Print paper-reported stats.")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.stats_only:
        print_reported_stats(config)
        return

    catalog_size = expected_catalog_size(k=config["topics"]["k"])
    expected_size = config["catalog"]["expected_size"]
    if catalog_size != expected_size:
        raise SystemExit(f"Catalog size mismatch: computed {catalog_size}, expected {expected_size}")

    if args.dry_run or (not args.all and not args.catalog):
        scenario_names = ", ".join(scenario.name for scenario in SCENARIOS)
        setting_names = ", ".join(config["settings"])
        print(f"Config OK: {args.config}")
        print(f"Scenarios: {scenario_names}")
        print(f"Settings: {setting_names}")
        print(f"p-hat catalog size: {catalog_size}")
        example_path = phat_user_path(
            "data/synthetic",
            "phat_driven_top2",
            "BLR",
            "pb_p",
            "train_pb_U1.json",
        )
        print(f"Example p-hat public path: {example_path}")
        print("Use --all to generate theta_driven and p_driven base logs.")
        print("Use scripts/generate_phat.py for p-hat-driven logs.")
        return

    if args.catalog:
        catalog_dir = args.output_root / "phat_item_catalog"
        cat_cfg = config.get("catalog", {})
        n = generate_catalog(
            output_dir=catalog_dir,
            k=config["topics"]["k"],
            seed=int(config.get("seed", 42)),
            samples_per_single=cat_cfg.get("one_topic", {}).get("samples_per_topic", 50),
            samples_per_pair=cat_cfg.get("two_topic", {}).get("samples_per_pair", 50),
            single_alpha_high=cat_cfg.get("one_topic", {}).get("alpha_high", 20.0),
            single_alpha_other=cat_cfg.get("one_topic", {}).get("alpha_other", 2.0),
            combo_alpha_high=cat_cfg.get("two_topic", {}).get("alpha_high", 20.0),
            combo_alpha_other=cat_cfg.get("two_topic", {}).get("alpha_other", 2.0),
            blend_alpha_high=cat_cfg.get("four_topic", {}).get("alpha_high", 8.0),
            blend_alpha_other=cat_cfg.get("four_topic", {}).get("alpha_other", 2.0),
        )
        print(f"Generated item catalog: {n} items under {catalog_dir}")

    if args.all:
        counts = write_theta_p_dataset(
            output_root=args.output_root,
            settings=args.settings,
            seed=int(config.get("seed", 42)),
        )
        print(
            "Generated theta/p synthetic data: "
            f"users={counts['users']}, interactions={counts['interactions']}, "
            f"p_changes={counts['preference_change_events']}"
        )
        print(f"Output root: {args.output_root}")
        print("Use scripts/02_generate_phat.py to derive p-hat-driven datasets from these base logs.")


if __name__ == "__main__":
    main()
