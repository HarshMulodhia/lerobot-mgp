from lerobot.configs import PreTrainedConfig
from lerobot.policies.mgp.configuration_mgp import MGPConfig


def test_mgp_config_is_registered():
    assert PreTrainedConfig.get_choice_class("mgp") is MGPConfig


def test_mgp_config_has_mgp_type():
    cfg = MGPConfig(device="cpu")
    assert cfg.type == "mgp"
