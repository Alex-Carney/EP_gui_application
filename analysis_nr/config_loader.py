import json
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    experiment_id: str
    colorplot_freq_min: float
    colorplot_freq_max: float
    cavity_freq_min: float
    cavity_freq_max: float
    yig_freq_min: float
    yig_freq_max: float
    current_min: float
    current_max: float
    db_path: str
    readout_type: str
    optimal_J: float
    optimal_J_unc: float
    simulated_vertical_offset: float


def load_config(config_path: str, config_name: str = "default") -> ExperimentConfig:
    """
    Load the configuration from a JSON file and return an ExperimentConfig
    dataclass for the given configuration name.
    """
    with open(config_path, "r") as file:
        all_configs = json.load(file)
    if config_name not in all_configs:
        raise KeyError(f"Configuration '{config_name}' not found in {config_path}.")
    cfg = all_configs[config_name]
    return ExperimentConfig(
        experiment_id=cfg["experiment_id"],
        colorplot_freq_min=cfg["frequency_limits"]["colorplot"]["min"],
        colorplot_freq_max=cfg["frequency_limits"]["colorplot"]["max"],
        cavity_freq_min=cfg["frequency_limits"]["cavity"]["min"],
        cavity_freq_max=cfg["frequency_limits"]["cavity"]["max"],
        yig_freq_min=cfg["frequency_limits"]["yig"]["min"],
        yig_freq_max=cfg["frequency_limits"]["yig"]["max"],
        current_min=cfg["current_limits"]["min"],
        current_max=cfg["current_limits"]["max"],
        db_path=cfg["db_path"],
        readout_type=cfg["readout_type"],
        optimal_J=cfg["optimal_J"],
        optimal_J_unc=cfg["optimal_J_unc"],
        simulated_vertical_offset=cfg["simulated_vertical_offset"],
    )
