"""Synthetic scenario metadata and theta/p-driven generation.

The original synthetic data was prototyped in notebooks under the
``generation`` directory.  This module keeps the notebook semantics in library
code so the public CLI can regenerate the theta-driven ("easy learning") and
p-driven ("difficult learning") artifacts with the compact JSON record schema
already used in ``data/synthetic``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ScenarioSpec:
    """Public description of one synthetic user-behavior scenario."""

    name: str
    title: str
    folder: str
    family: str
    two_topic: bool = False
    conservation: bool = False


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("PS", "Preference Shifting", "ps", "shifting"),
    ScenarioSpec(
        "PSC1T",
        "Preference Shifting and Conservation",
        "psc1t",
        "shifting",
        conservation=True,
    ),
    ScenarioSpec(
        "PSC2T",
        "Preference Shifting and Conservation with Two Topics",
        "psc2t",
        "shifting",
        two_topic=True,
        conservation=True,
    ),
    ScenarioSpec("PB", "Preference Broadening", "pb", "broadening"),
    ScenarioSpec(
        "PBC1T",
        "Preference Broadening and Conservation",
        "pbc1t",
        "broadening",
        conservation=True,
    ),
    ScenarioSpec(
        "PBC2T",
        "Preference Broadening and Conservation with Two Topics",
        "pbc2t",
        "broadening",
        two_topic=True,
        conservation=True,
    ),
)

SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
SCENARIO_BY_FOLDER = {scenario.folder: scenario for scenario in SCENARIOS}

SETTING_OUTPUT_DIR = {
    "theta_driven": "theta_driven",
    "p_driven": "p_driven",
}

SETTING_KIND = {
    "theta_driven": "theta",
    "p_driven": "p",
}

TRAIN_USERS: Mapping[str, Mapping[str, Sequence[int]]] = {
    "theta_driven": {
        "ps": (11, 13, 2, 4, 8),
        "psc1t": (1, 11, 3, 5, 9),
        "psc2t": (1, 10, 12, 13, 14),
        "pb": (1, 10, 12, 3, 7),
        "pbc1t": (1, 13, 14, 4, 8),
        "pbc2t": (12, 13, 14, 5, 9),
    },
    "p_driven": {
        "ps": (1, 13, 5, 8, 9),
        "psc1t": (1, 10, 2, 3, 7),
        "psc2t": (1, 12, 3, 8, 9),
        "pb": (1, 10, 13, 5, 6),
        "pbc1t": (12, 2, 4, 5, 6),
        "pbc2t": (1, 10, 12, 4, 6),
    },
}

# These timeline lengths mirror the committed synthetic artifact.  The missing
# p-driven PBC1T U4 receives the length needed to recover the paper Table 1
# interaction and p-change totals.
USER_LENGTHS: Mapping[str, Mapping[str, Mapping[int, int]]] = {
    "theta_driven": {
        "ps": {1: 40, 2: 45, 3: 50, 4: 55, 5: 60, 6: 65, 7: 70, 8: 53, 9: 58, 10: 48, 11: 35, 12: 68, 13: 44, 14: 57, 15: 30},
        "psc1t": {1: 30, 2: 40, 3: 50, 4: 60, 5: 70, 6: 40, 7: 50, 8: 60, 9: 70, 10: 50, 11: 60, 12: 70, 13: 50, 14: 60, 15: 70},
        "psc2t": {1: 80, 2: 90, 3: 100, 4: 110, 5: 130, 6: 70, 7: 80, 8: 90, 9: 100, 10: 110, 11: 70, 12: 80, 13: 90, 14: 100, 15: 110},
        "pb": {1: 30, 2: 33, 3: 38, 4: 40, 5: 45, 6: 48, 7: 51, 8: 55, 9: 58, 10: 62, 11: 65, 12: 70, 13: 57, 14: 50, 15: 46},
        "pbc1t": {1: 30, 2: 40, 3: 50, 4: 60, 5: 70, 6: 50, 7: 60, 8: 70, 9: 70, 10: 50, 11: 50, 12: 70, 13: 70, 14: 70, 15: 70},
        "pbc2t": {1: 70, 2: 80, 3: 90, 4: 100, 5: 110, 6: 70, 7: 80, 8: 90, 9: 100, 10: 110, 11: 70, 12: 80, 13: 90, 14: 100, 15: 110},
    },
    "p_driven": {
        "ps": {1: 30, 2: 35, 3: 38, 4: 40, 5: 43, 6: 45, 7: 48, 8: 50, 9: 52, 10: 55, 11: 57, 12: 60, 13: 65, 14: 70, 15: 68},
        "psc1t": {1: 30, 2: 40, 3: 50, 4: 60, 5: 70, 6: 50, 7: 60, 8: 70, 9: 50, 10: 60, 11: 70, 12: 50, 13: 60, 14: 70, 15: 70},
        "psc2t": {1: 70, 2: 80, 3: 90, 4: 100, 5: 110, 6: 70, 7: 80, 8: 90, 9: 100, 10: 110, 11: 70, 12: 80, 13: 90, 14: 100, 15: 110},
        "pb": {1: 30, 2: 35, 3: 40, 4: 45, 5: 48, 6: 50, 7: 52, 8: 58, 9: 60, 10: 61, 11: 65, 12: 68, 13: 70, 14: 57, 15: 47},
        "pbc1t": {1: 30, 2: 40, 3: 50, 4: 124, 5: 60, 6: 70, 7: 50, 8: 60, 9: 70, 10: 50, 11: 70, 12: 40, 13: 50, 14: 60, 15: 70},
        "pbc2t": {1: 70, 2: 80, 3: 90, 4: 100, 5: 110, 6: 70, 7: 80, 8: 90, 9: 100, 10: 110, 11: 70, 12: 80, 13: 90, 14: 100, 15: 110},
    },
}

REACTION_PATTERNS: Mapping[str, Sequence[str]] = {
    "PSC1T": ("neutral", "like", "hate", "hate", "like", "neutral", "like", "hate"),
    "PBC1T": ("hate", "like", "neutral", "hate", "like", "neutral", "like", "hate"),
    "PSC2T": ("neutral", "hate", "like", "neutral", "two_like", "hate", "two_like", "two_like", "hate", "neutral"),
    "PBC2T": ("neutral", "like", "like", "neutral", "two_like", "hate", "two_like", "neutral", "hate", "two_like"),
}


def validate_scenario_name(name: str) -> str:
    """Return a normalized scenario name or raise a useful error."""

    normalized = name.upper()
    if normalized not in SCENARIO_BY_NAME:
        valid = ", ".join(sorted(SCENARIO_BY_NAME))
        raise ValueError(f"Unknown scenario {name!r}. Expected one of: {valid}")
    return normalized


def scenario_folder(name: str) -> str:
    """Return the public lower-case folder name for a paper scenario."""

    return SCENARIO_BY_NAME[validate_scenario_name(name)].folder


def change_steps(total_steps: int, first_change: int = 11, cycle_length: int = 10) -> List[int]:
    """Return candidate p-change steps in one-based indexing."""

    return list(range(first_change, total_steps + 1, cycle_length))


def split_for_user(setting: str, scenario_folder_name: str, user_id: int) -> str:
    train_users = set(TRAIN_USERS[setting][scenario_folder_name])
    return "train" if user_id in train_users else "test"


def _boost(value: float, eta: float) -> float:
    return value + eta * (1.0 - value)


def _downboost(value: float, eta: float) -> float:
    return value - eta * (1.0 + value)


def _add_step_noise(
    p: np.ndarray,
    alpha_t: np.ndarray,
    rng: np.random.Generator,
    seen_noise_sigma: float,
    trivial_noise_sigma: float,
) -> None:
    seen_mask = alpha_t >= 2.0
    if np.any(seen_mask):
        idx = np.where(seen_mask)[0]
        p[idx] = np.clip(
            p[idx] + rng.normal(0.0, seen_noise_sigma, size=idx.size),
            -1.0,
            1.0,
        )
    if np.any(~seen_mask):
        idx = np.where(~seen_mask)[0]
        p[idx] = np.clip(
            p[idx] + rng.normal(0.0, trivial_noise_sigma, size=idx.size),
            -0.01,
            0.01,
        )


def _select_normal_high_topics(
    setting_kind: str,
    seen: Iterable[int],
    p: np.ndarray,
    rng: np.random.Generator,
) -> List[int]:
    if setting_kind == "theta":
        candidates = sorted(seen)
    elif setting_kind == "p":
        candidates = list(np.argsort(-p)[: min(3, p.size)])
    else:
        raise ValueError(f"Unknown setting kind: {setting_kind}")

    if len(candidates) <= 1:
        return candidates
    return [int(x) for x in rng.choice(candidates, size=2, replace=False)]


def _assign_high_alphas(alpha_t: np.ndarray, topics: Sequence[int], rng: np.random.Generator) -> None:
    if not topics:
        return
    if len(topics) == 1:
        alpha_t[topics[0]] = 20.0
        return
    highs = [15.0, 20.0]
    rng.shuffle(highs)
    for topic, high in zip(topics[:2], highs):
        alpha_t[topic] = high


def _next_new_topic(seen: Iterable[int], k_topics: int) -> Optional[int]:
    seen_set = set(seen)
    for topic in range(k_topics):
        if topic not in seen_set:
            return topic
    return None


def _low_preference_pair(
    p: np.ndarray,
    seen: Iterable[int],
    exclude: Iterable[int] = (),
) -> List[int]:
    excluded = set(exclude)
    candidates = [topic for topic in seen if topic not in excluded]
    if len(candidates) < 2:
        candidates = [topic for topic in range(p.size) if topic not in excluded]
    candidates = sorted(candidates, key=lambda topic: (p[topic], topic))
    return candidates[:2]


def _event_actions(
    scenario: ScenarioSpec,
    event_index: int,
    seen: Iterable[int],
    p: np.ndarray,
    k_topics: int,
    user_id: int,
) -> List[Tuple[int, str]]:
    """Return zero-based event topics and actions for one candidate change step."""

    if not scenario.conservation:
        topic = _next_new_topic(seen, k_topics)
        if topic is None:
            topic = _low_preference_pair(p, range(k_topics))[0]
        return [(topic, "like")]

    pattern = REACTION_PATTERNS[scenario.name]
    action = pattern[(event_index + user_id - 1) % len(pattern)]

    if scenario.two_topic and action == "two_like":
        pair = _low_preference_pair(p, seen)
        return [(pair[0], "like"), (pair[1], "like")]

    topic = _next_new_topic(seen, k_topics)
    if topic is None:
        topic = _low_preference_pair(p, seen)[0]
    return [(topic, action)]


def _apply_actions(
    p: np.ndarray,
    actions: Sequence[Tuple[int, str]],
    seen_before: Iterable[int],
    shift_existing_preferences: bool,
    rng: np.random.Generator,
    eta_big: float,
    neutral_sigma: float,
) -> None:
    liked: List[Tuple[int, float]] = []

    for topic, action in actions:
        if action == "like":
            gain = eta_big * (1.0 - p[topic])
            p[topic] += gain
            liked.append((topic, gain))
        elif action == "hate":
            p[topic] = _downboost(p[topic], eta_big)
        elif action == "neutral":
            p[topic] = np.clip(p[topic] + rng.normal(0.0, neutral_sigma), -1.0, 1.0)
        else:
            raise ValueError(f"Unknown reaction action: {action}")

    if not shift_existing_preferences or not liked:
        return

    liked_topics = {topic for topic, _ in liked}
    total_gain = sum(gain for _, gain in liked)
    others = np.array(
        sorted(topic for topic in seen_before if topic not in liked_topics),
        dtype=int,
    )
    if others.size == 0:
        return

    z = float(np.sum(p[others] + 1.0))
    if z > 0.0:
        p[others] = p[others] - ((p[others] + 1.0) / z) * total_gain


def _round_vector(values: np.ndarray, decimals: int = 3) -> List[float]:
    return [round(float(x), decimals) for x in values]


def generate_timeline(
    setting: str,
    scenario_name: str,
    total_steps: int,
    seed: int,
    k_topics: int = 10,
    initial_explored: int = 4,
    eta_big: float = 0.8,
    eta_small: float = 0.3,
    seen_noise_sigma: float = 0.02,
    trivial_noise_sigma: float = 0.002,
    neutral_sigma: float = 0.05,
    high_topic_threshold: float = 0.35,
    change_label_l2_threshold: float = 0.25,
) -> List[dict]:
    """Generate one theta- or p-driven user timeline.

    The returned records intentionally match the current public JSON files:
    a bare list of timestep dictionaries with ``topic_vector``, ``rating``,
    ``preference_vector``, and ``preference_change_label``.
    """

    if setting not in SETTING_KIND:
        raise ValueError(f"Unknown setting {setting!r}. Expected one of {sorted(SETTING_KIND)}")
    scenario = SCENARIO_BY_NAME[validate_scenario_name(scenario_name)]
    setting_kind = SETTING_KIND[setting]
    rng = np.random.default_rng(seed)

    p = np.zeros(k_topics, dtype=float)
    seen = set(range(initial_explored))
    records: List[dict] = []
    previous_p: Optional[np.ndarray] = None
    event_steps = set(change_steps(total_steps))
    event_count = 0

    for timestep in range(1, total_steps + 1):
        seen_before = set(seen)
        actions: List[Tuple[int, str]] = []
        if timestep in event_steps:
            actions = _event_actions(
                scenario,
                event_count,
                seen_before,
                p,
                k_topics,
                seed,
            )
            event_count += 1

        alpha_t = np.ones(k_topics, dtype=float)
        for topic in seen:
            alpha_t[topic] = 2.0

        if actions:
            for topic, _ in actions:
                seen.add(topic)
            for topic in seen:
                alpha_t[topic] = 2.0
            for topic, _ in actions:
                alpha_t[topic] = 20.0
        else:
            high_topics = _select_normal_high_topics(setting_kind, seen, p, rng)
            _assign_high_alphas(alpha_t, high_topics, rng)

        theta_t = rng.dirichlet(alpha_t)

        if timestep == 1:
            high_seen = [topic for topic in seen if alpha_t[topic] >= 15.0]
            base_seen = [topic for topic in seen if 2.0 <= alpha_t[topic] < 15.0]
            for topic in high_seen:
                p[topic] = _boost(p[topic], eta_big)
            for topic in base_seen:
                p[topic] = _boost(p[topic], eta_small)
            unseen = [topic for topic in range(k_topics) if topic not in seen]
            if unseen:
                p[unseen] = rng.uniform(-0.01, 0.01, size=len(unseen))
        elif actions:
            if not scenario.conservation:
                actions = [
                    (topic, action)
                    for topic, action in actions
                    if theta_t[topic] >= high_topic_threshold
                ]
            _apply_actions(
                p,
                actions,
                seen_before,
                shift_existing_preferences=(scenario.family == "shifting"),
                rng=rng,
                eta_big=eta_big,
                neutral_sigma=neutral_sigma,
            )

        _add_step_noise(p, alpha_t, rng, seen_noise_sigma, trivial_noise_sigma)
        seen |= {topic for topic in range(k_topics) if alpha_t[topic] >= 2.0}

        if previous_p is None:
            label = 0
        else:
            label = int(float(np.linalg.norm(p - previous_p)) > change_label_l2_threshold)
        previous_p = p.copy()

        records.append(
            {
                "topic_vector": _round_vector(theta_t),
                "rating": round(float(np.dot(p, theta_t)), 3),
                "preference_vector": _round_vector(p),
                "preference_change_label": label,
            }
        )

    return records


def user_plan(settings: Optional[Iterable[str]] = None) -> List[Tuple[str, str, int, str, int]]:
    """Return (setting, scenario_folder, user_id, split, total_steps)."""

    requested = list(settings) if settings is not None else list(SETTING_KIND)
    plan: List[Tuple[str, str, int, str, int]] = []
    for setting in requested:
        if setting not in USER_LENGTHS:
            raise ValueError(f"Unknown generated setting: {setting}")
        for scenario in SCENARIOS:
            folder = scenario.folder
            for user_id, total_steps in sorted(USER_LENGTHS[setting][folder].items()):
                plan.append(
                    (
                        setting,
                        folder,
                        user_id,
                        split_for_user(setting, folder, user_id),
                        total_steps,
                    )
                )
    return plan


def write_theta_p_dataset(
    output_root: Path,
    settings: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> Dict[str, int]:
    """Generate theta-driven and/or p-driven JSON files under ``output_root``."""

    counts = {"users": 0, "interactions": 0, "preference_change_events": 0}
    for setting, folder, user_id, split, total_steps in user_plan(settings):
        scenario = SCENARIO_BY_FOLDER[folder]
        user_seed = seed + (1994 if setting == "p_driven" else 0) + user_id * 97
        user_seed += list(SCENARIO_BY_FOLDER).index(folder) * 1009
        records = generate_timeline(
            setting=setting,
            scenario_name=scenario.name,
            total_steps=total_steps,
            seed=user_seed,
        )
        path = output_root / SETTING_OUTPUT_DIR[setting] / split / folder / f"{folder}_U{user_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, separators=(",", ":"))

        counts["users"] += 1
        counts["interactions"] += len(records)
        counts["preference_change_events"] += sum(
            int(record["preference_change_label"]) for record in records
        )
    return counts
