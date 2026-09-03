"""Scaffold smoke tests: vocabularies are fixed and settings load without a .env file."""

from recoverai import config


def test_exactly_five_arms() -> None:
    assert len(config.ARMS) == 5
    assert config.ARMS[config.NO_ACTION] == "no_action"
    assert "discount" not in " ".join(config.ARMS)


def test_failure_mix_sums_to_one() -> None:
    assert abs(sum(config.FAILURE_MIX.values()) - 1.0) < 1e-9
    assert set(config.FAILURE_MIX) == set(config.FAILURE_CATEGORIES) == set(config.FAILURE_SOURCE)


def test_settings_defaults(tmp_path) -> None:
    s = config.get_settings(env_file=tmp_path / "missing.env")
    assert s.seed == 42 and s.sim_variant == "calibrated" and s.executor == "mock"


def test_dotenv_parsing(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text('RECOVERAI_SEED=7\nRECOVERAI_EXECUTOR_FAILURE_RATE=0.2  # comment\n# x\n')
    s = config.get_settings(env_file=env)
    assert s.seed == 7 and s.executor_failure_rate == 0.2
