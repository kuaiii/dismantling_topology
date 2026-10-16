import pytest
from importlib.resources import files

from autosnd.config import api_environment, load_config


def test_paper_budget_values():
    config = load_config("configs/paper.yaml")
    assert config.stage1.candidates == 300
    assert config.stage2.proposals == 10
    assert config.stage3.candidates == 200
    assert config.stage3.seed_quotas == {
        "quality": 8,
        "speed": 6,
        "balanced": 6,
        "repair": 4,
    }
    assert config.stage3.branch_budgets == {
        "quality": 60,
        "speed": 60,
        "balanced": 50,
        "repair": 30,
    }


def test_api_credentials_are_required_from_environment(monkeypatch):
    monkeypatch.delenv("AUTOSND_API_URL", raising=False)
    monkeypatch.delenv("AUTOSND_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AUTOSND_API_URL"):
        api_environment()


def test_root_and_packaged_default_configs_are_identical():
    for name in ("smoke.yaml", "paper.yaml"):
        root = ("configs/" + name)
        assert open(root, encoding="utf-8").read() == (
            files("autosnd.defaults").joinpath(name).read_text(encoding="utf-8")
        )


def test_packaged_default_loads_outside_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = load_config("configs/smoke.yaml")
    assert config.stage1.candidates == 4
    assert config.stage3.candidates == 4
