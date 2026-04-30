from nspb.catalog import expected_catalog_size
from nspb.hyperparameters import load_hyperparameter_group
from nspb.paths import phat_user_path, public_user_filename
from nspb.scenarios import (
    SCENARIO_BY_NAME,
    generate_timeline,
    user_plan,
    validate_scenario_name,
    write_theta_p_dataset,
)

import numpy as np
import pytest


def test_expected_phat_catalog_size():
    assert expected_catalog_size(k=10) == 2960


def test_paper_scenarios_are_registered():
    assert set(SCENARIO_BY_NAME) == {"PS", "PSC1T", "PSC2T", "PB", "PBC1T", "PBC2T"}


def test_validate_scenario_name_normalizes_case():
    assert validate_scenario_name("ps") == "PS"


def test_public_user_filename_removes_split_prefix():
    assert public_user_filename("train__ps_U1.json") == "ps_U1.json"
    assert public_user_filename("test__pb_U11.json") == "pb_U11.json"
    assert public_user_filename("train_pb_U1.json") == "pb_U1.json"
    assert public_user_filename("test_pb_U11.json") == "pb_U11.json"
    assert public_user_filename("ps_U2.json") == "ps_U2.json"


def test_phat_user_path_has_no_all_layer_and_no_split_prefix():
    path = phat_user_path(
        "data/synthetic",
        "phat_driven_top2",
        "BLR",
        "pb_p",
        "train_pb_U1.json",
    )

    assert str(path) == "data/synthetic/phat_driven_top2/BLR/pb_p/pb_U1.json"
    assert "/all/" not in str(path)


def test_model_hyperparameters_are_configured():
    evaluation = load_hyperparameter_group("evaluation")
    phat_generation = load_hyperparameter_group("phat_generation")

    assert set(evaluation) == {"KF", "AROW", "BLR", "vbBLR", "fBLR", "BLRsw", "PBLR", "NIG"}
    assert set(phat_generation) == {"KF", "AROW", "BLR", "vbBLR", "fBLR", "BLRsw", "PBLR", "NIW"}


def test_theta_p_generator_uses_current_json_record_schema():
    records = generate_timeline("theta_driven", "PS", total_steps=12, seed=7)

    assert len(records) == 12
    assert set(records[0]) == {
        "topic_vector",
        "rating",
        "preference_vector",
        "preference_change_label",
    }
    assert len(records[0]["topic_vector"]) == 10
    assert len(records[0]["preference_vector"]) == 10
    assert sum(records[0]["topic_vector"]) == pytest.approx(1.0, abs=1e-3)


def test_theta_p_generator_change_labels_follow_l2_threshold():
    records = generate_timeline("p_driven", "PBC2T", total_steps=45, seed=19)
    prefs = [np.asarray(record["preference_vector"], dtype=float) for record in records]

    expected = [0]
    for current, previous in zip(prefs[1:], prefs[:-1]):
        expected.append(int(float(np.linalg.norm(current - previous)) > 0.25))

    assert [record["preference_change_label"] for record in records] == expected


def test_theta_p_user_plan_has_paper_base_setting_user_counts():
    plan = user_plan(["theta_driven", "p_driven"])

    assert len(plan) == 180
    assert sum(1 for item in plan if item[0] == "theta_driven") == 90
    assert sum(1 for item in plan if item[0] == "p_driven") == 90


def test_write_theta_p_dataset_writes_expected_layout(tmp_path):
    counts = write_theta_p_dataset(tmp_path, settings=["p_driven"], seed=42)

    assert counts["users"] == 90
    assert (tmp_path / "p_driven" / "train" / "pbc1t" / "pbc1t_U4.json").is_file()
