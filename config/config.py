# config.py

# Drivers path
DRIVERS_PATH = "S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers"

# DRIVERS_PATH = r"E:\Dartmouth College Dropbox\Dartmouth College Dropbox\Alexander Carney\instrument_drivers"

# Serial numbers for switches
DRIVE_SWITCH_SERIAL = 35353
READOUT_SWITCH_SERIAL = 34875

CAV_TO_YIG_SWITCH_SERIAL = 35096
YIG_TO_CAV_SWITCH_SERIAL = 35095

DRIVE_YIG_SERIAL = 35353
DRIVE_CAVITY_SERIAL = 35095
READOUT_YIG_SERIAL = 35096
READOUT_CAVITY_SERIAL = 34875

DRIVE_YIG_CONFIG = {
    "ENABLE": 3,
    "DISABLE": 4
}

DRIVE_CAVITY_CONFIG = {
    "ENABLE": 2,
    "DISABLE": 1
}

READOUT_YIG_CONFIG = {
    "ENABLE": 1,
    "DISABLE": 2
}

READOUT_CAVITY_CONFIG = {
    "ENABLE": 4,
    "DISABLE": 1
}

# Center frequency
CENTER_FREQ = 6.0e9

# Voltage controller settings
VOLTAGE_CONTROLLER_PORT = "COM12"
VOLTAGE_CONTROLLER_BAUDRATE = 115200

# Default step sizes for voltage control
VOLTAGE_STEP_SIZES = [1.0, 0.1, 0.01, 0.001]

# Step sizes for phase shifters in degrees
PHASE_SHIFTER_STEP_SIZES = [1, 5, 10]

# Step sizes for attenuators in dB
ATTENUATOR_STEP_SIZES = [0.1, 0.5, 1.0]

# Define phase shifter devices with their serial numbers
PHASE_SHIFTER_DEVICES = [
    {"name": "loop_phase", "serial": 35431},
    {"name": "yig_phase", "serial": 31093},
    {"name": "cavity_phase", "serial": 32351},
]

# Define attenuator devices with their serial numbers
ATTENUATOR_DEVICES = [
    {"name": "cavity_att", "serial": 33044},
    {"name": "yig_att", "serial": 28578},
]

# VNA default values
VNA_DEFAULT_CENTER_FREQ = 6.0e9
VNA_DEFAULT_SPAN_FREQ = 40e6
VNA_DEFAULT_ATTENUATION = 20
VNA_DEFAULT_DEL_F = 0.2e6

# VNA LO serial number and other settings
VNA_LO_SERIAL_NUMBER = "10002F1B"
VNA_BASE_DIR = "results_gui"
VNA_N_AVG = 15  # For faster updates

# Loop attenuation settings
LOOP_ATT = 5
LOOP_ATT_BACK_OFFSET = 4.5

# YIG drive - are we allowing the YIG to be readout?
YIG_DRIVE_ATTEN_HIGH = 50
YIG_DRIVE_ATTEN_LOW = 0

MODE = "NR"  # "PT" or "NR"

# SWITCH_35353_CONFIG = {
#     "CAVITY": 1,
#     "YIG": 0,
#     "NULL": 4
# }
#
# SWITCH_34875_CONFIG = {
#     "CAVITY": 0,
#     "YIG": 3,
#     "NULL": 1
# }
#
# LOOP_35095_CONFIG = {
#     "ON": 2,
#     "NULL": 1
# }

DRIVE_SWITCH_CONFIG = {
    "CAVITY": 1,
    "YIG": 3,
}

READOUT_SWITCH_CONFIG = {
    "CAVITY": 4,
    "YIG": 2,
}
