import json
from dataclasses import dataclass


@dataclass
class PTExperimentConfig:
    experiment_id: str
    colorplot_freq_min: float
    colorplot_freq_max: float
    cavity_freq_min: float
    cavity_freq_max: float
    yig_freq_min: float
    yig_freq_max: float
    voltage_min: float
    voltage_max: float
    db_path: str
    readout_type: str
    optimal_J: float
    optimal_J_unc: float
    simulated_vertical_offset: float
    amplitude_threshold_overfitting: float
    overlap_region_start: float
    overlap_region_end: float


def load_config(config_path: str, config_name: str = "default") -> PTExperimentConfig:
    """
    Load the configuration from a JSON file and return an ExperimentConfig
    dataclass for the given configuration name.
    """
    with open(config_path, "r") as file:
        all_configs = json.load(file)
    if config_name not in all_configs:
        raise KeyError(f"Configuration '{config_name}' not found in {config_path}.")
    cfg = all_configs[config_name]
    return PTExperimentConfig(
        experiment_id=cfg["experiment_id"],
        colorplot_freq_min=cfg["frequency_limits"]["colorplot"]["min"],
        colorplot_freq_max=cfg["frequency_limits"]["colorplot"]["max"],
        cavity_freq_min=cfg["frequency_limits"]["cavity"]["min"],
        cavity_freq_max=cfg["frequency_limits"]["cavity"]["max"],
        yig_freq_min=cfg["frequency_limits"]["yig"]["min"],
        yig_freq_max=cfg["frequency_limits"]["yig"]["max"],
        voltage_min=cfg["voltage_limits"]["min"],
        voltage_max=cfg["voltage_limits"]["max"],
        db_path=cfg["db_path"],
        readout_type=cfg["readout_type"],
        optimal_J=cfg["optimal_J"],
        optimal_J_unc=cfg["optimal_J_unc"],
        simulated_vertical_offset=cfg["simulated_vertical_offset"],
        amplitude_threshold_overfitting=cfg["amplitude_threshold_overfitting"],
        overlap_region_start=cfg["overlap_region_start"] if "overlap_region_start" in cfg else None,
        overlap_region_end=cfg["overlap_region_end"] if "overlap_region_end" in cfg else None,
    )
