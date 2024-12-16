# config.py

# Drivers path
DRIVERS_PATH = "S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers"

# Serial numbers for switches
DRIVE_SWITCH_SERIAL = 34875
READOUT_SWITCH_SERIAL = 35353

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
VNA_DEFAULT_CENTER_FREQ = 6.005e9
VNA_DEFAULT_SPAN_FREQ = 20e6
VNA_DEFAULT_ATTENUATION = 20
VNA_DEFAULT_DEL_F = 0.01e6

# VNA LO serial number and other settings
VNA_LO_SERIAL_NUMBER = "10002F1B"
VNA_BASE_DIR = "results_gui"
VNA_N_AVG = 15  # For faster updates

# Loop attenuation settings
LOOP_ATT = 30
LOOP_ATT_BACK_OFFSET = 4.5

# YIG drive - are we allowing the YIG to be readout?
YIG_DRIVE_ATTEN_HIGH = 50
YIG_DRIVE_ATTEN_LOW = 0

