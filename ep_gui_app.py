import sys
import threading
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QMessageBox, QStatusBar, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import configuration
import app_config as config

# Add drivers path
if config.DRIVERS_PATH not in sys.path:
    sys.path.append(config.DRIVERS_PATH)

# Import necessary modules
from components.dc205 import SRSDC205 as dc
from microwave_switch import MicrowaveSwitchController
from sc5511a import *
from LDA import Vaunix_LDA
from LPS import Vaunix_LPS
from vna.vna_instrument import VectorNetworkAnalyzer

# Import find_peaks from scipy.signal
from scipy.signal import find_peaks

# DeviceController for Voltage Control
class DeviceController:
    def __init__(self, mock=False):
        self.voltage = 0.0  # Initial voltage
        if mock:
            self.inst = None
        else:
            self.inst = dc.open_serial(config.VOLTAGE_CONTROLLER_PORT, config.VOLTAGE_CONTROLLER_BAUDRATE)
        
    def set_voltage(self, voltage):
        self.voltage = round(voltage, 8)
        if self.inst:
            self.inst.channel[0].voltage = voltage
        else:
            print(f"Voltage set to {voltage} V")
        
    def get_voltage(self):
        return self.voltage

# Voltage Control Panel
class VoltageControlPanel(QWidget):
    def __init__(self, parent=None, device_controller=None):
        super().__init__(parent)
        self.device_controller = device_controller or DeviceController(mock=False)
        self.step_size = config.VOLTAGE_STEP_SIZES[0]  # Default step size is 1V

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Voltage display
        self.voltage_label = QLabel(f"Current Voltage: {self.device_controller.get_voltage()} V")
        layout.addWidget(self.voltage_label)

        # Increase and decrease buttons
        button_layout = QHBoxLayout()
        self.increase_button = QPushButton("↑")
        self.increase_button.clicked.connect(self.increase_voltage)
        button_layout.addWidget(self.increase_button)

        self.decrease_button = QPushButton("↓")
        self.decrease_button.clicked.connect(self.decrease_voltage)
        button_layout.addWidget(self.decrease_button)
        layout.addLayout(button_layout)

        # Entry to manually set voltage
        self.voltage_entry = QLineEdit()
        layout.addWidget(self.voltage_entry)

        # Set voltage button
        self.set_button = QPushButton("Set Voltage")
        self.set_button.clicked.connect(self.set_voltage)
        layout.addWidget(self.set_button)

        # Step size selection
        step_layout = QHBoxLayout()
        self.step_label = QLabel("Step Size:")
        step_layout.addWidget(self.step_label)

        self.step_combo = QComboBox()
        self.step_combo.addItems([str(s) for s in config.VOLTAGE_STEP_SIZES])
        self.step_combo.currentIndexChanged.connect(self.update_step_size)
        step_layout.addWidget(self.step_combo)
        layout.addLayout(step_layout)

        self.setLayout(layout)

    def update_step_size(self):
        self.step_size = float(self.step_combo.currentText())

    def update_voltage_display(self):
        self.voltage_label.setText(f"Current Voltage: {self.device_controller.get_voltage()} V")

    def increase_voltage(self):
        current_voltage = self.device_controller.get_voltage()
        self.device_controller.set_voltage(current_voltage + self.step_size)
        self.update_voltage_display()

    def decrease_voltage(self):
        current_voltage = self.device_controller.get_voltage()
        self.device_controller.set_voltage(current_voltage - self.step_size)
        self.update_voltage_display()

    def set_voltage(self):
        try:
            new_voltage = float(self.voltage_entry.text())
            self.device_controller.set_voltage(new_voltage)
            self.update_voltage_display()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")

# Switch Configuration Panel
class SwitchConfigurationPanel(QWidget):
    configuration_changed = pyqtSignal(str)  # Signal to emit when configuration changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.switch_controller = MicrowaveSwitchController()

        # Initialize attenuators
        self.loop_atten = Vaunix_LDA("LDA", 28577, dll_path=config.DRIVERS_PATH, test_mode=False)
        self.loop_atten_back = Vaunix_LDA("LDA4", 34907, dll_path=config.DRIVERS_PATH, test_mode=False)
        self.yig_drive_atten = Vaunix_LDA("LDA2", 33291, dll_path=config.DRIVERS_PATH, test_mode=False)

        # Setup working frequencies
        self.loop_atten.working_frequency(config.CENTER_FREQ)
        self.yig_drive_atten.working_frequency(config.CENTER_FREQ)
        self.loop_atten_back.working_frequency(config.CENTER_FREQ)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Label
        label = QLabel("Switch Configuration")
        layout.addWidget(label)

        # Buttons for each mode
        self.normal_button = QPushButton("Normal Operation")
        self.normal_button.clicked.connect(self.set_normal_operation)
        layout.addWidget(self.normal_button)

        self.cavity_button = QPushButton("Cavity Readout Only")
        self.cavity_button.clicked.connect(self.set_cavity_readout)
        layout.addWidget(self.cavity_button)

        self.yig_button = QPushButton("YIG Readout Only")
        self.yig_button.clicked.connect(self.set_yig_readout)
        layout.addWidget(self.yig_button)

        self.status_label = QLabel("Current Configuration: None")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def set_both_switches(self, port):
        try:
            self.switch_controller.set_active_port(port, config.DRIVE_SWITCH_SERIAL)
            self.switch_controller.set_active_port(port, config.READOUT_SWITCH_SERIAL)
            print(f"Both switches set to port {port}")
        except Exception as e:
            print(f"Error setting switches: {e}")

    def set_normal_operation(self):
        print("Normal operation selected.")
        LOOP_ATT = config.LOOP_ATT
        self.loop_atten.attenuation(LOOP_ATT)
        self.loop_atten_back.attenuation(LOOP_ATT + config.LOOP_ATT_BACK_OFFSET)
        self.yig_drive_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.set_both_switches(1)
        self.status_label.setText("Current Configuration: Normal Operation")
        self.configuration_changed.emit("Normal Operation")  # Emit signal

    def set_cavity_readout(self):
        print("Cavity readout only selected.")
        self.loop_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.loop_atten_back.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.yig_drive_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.set_both_switches(1)
        self.status_label.setText("Current Configuration: Cavity Readout Only")
        self.configuration_changed.emit("Cavity Readout Only")  # Emit signal

    def set_yig_readout(self):
        print("YIG readout only selected.")
        self.loop_atten.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.loop_atten_back.attenuation(config.YIG_DRIVE_ATTEN_HIGH)
        self.yig_drive_atten.attenuation(config.YIG_DRIVE_ATTEN_LOW)
        self.set_both_switches(2)
        self.status_label.setText("Current Configuration: YIG Readout Only")
        self.configuration_changed.emit("YIG Readout Only")  # Emit signal

# Phase Shifter Panel
class PhaseShifterPanel(QWidget):
    def __init__(self, parent, name, device_serial):
        super().__init__(parent)
        self.name = name
        self.device_serial = device_serial
        self.step_size = config.PHASE_SHIFTER_STEP_SIZES[0]
        self.value = 0  # Will be updated from device

        # Initialize the device
        self.device = Vaunix_LPS(name, device_serial, dll_path=config.DRIVERS_PATH, test_mode=False)
        self.device.working_frequency(config.CENTER_FREQ)

        # Read the current value from the device
        try:
            self.value = round(self.device.phase.get_raw(), 1)
        except Exception as e:
            print(f"Error reading initial value for {self.name}: {e}")
            self.value = 0  # Default value if read fails

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel(self.name)
        layout.addWidget(self.label)

        self.value_label = QLabel(f"Current Value: {self.value}°")
        layout.addWidget(self.value_label)

        button_layout = QHBoxLayout()
        self.increase_button = QPushButton("↑")
        self.increase_button.clicked.connect(self.increase_value)
        button_layout.addWidget(self.increase_button)

        self.decrease_button = QPushButton("↓")
        self.decrease_button.clicked.connect(self.decrease_value)
        button_layout.addWidget(self.decrease_button)
        layout.addLayout(button_layout)

        # Entry to manually set value
        self.value_entry = QLineEdit()
        layout.addWidget(self.value_entry)

        # Set value button
        self.set_button = QPushButton("Set Value")
        self.set_button.clicked.connect(self.set_value)
        layout.addWidget(self.set_button)

        # Step size selection
        step_layout = QHBoxLayout()
        self.step_label = QLabel("Step Size:")
        step_layout.addWidget(self.step_label)

        self.step_combo = QComboBox()
        self.step_combo.addItems([str(s) for s in config.PHASE_SHIFTER_STEP_SIZES])
        self.step_combo.currentIndexChanged.connect(self.update_step_size)
        step_layout.addWidget(self.step_combo)
        layout.addLayout(step_layout)

        self.setLayout(layout)

    def update_step_size(self):
        self.step_size = float(self.step_combo.currentText())

    def update_value_display(self):
        self.value_label.setText(f"Current Value: {self.value}°")

    def increase_value(self):
        new_value = (self.value + self.step_size) % 360
        self.set_device_value(new_value)

    def decrease_value(self):
        new_value = (self.value - self.step_size) % 360
        self.set_device_value(new_value)

    def set_value(self):
        try:
            new_value = float(self.value_entry.text()) % 360
            self.set_device_value(new_value)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")

    def set_device_value(self, new_value):
        # Update the device value
        value_rad = np.deg2rad(new_value)
        self.device.phase(value_rad)
        self.value = round(new_value, 2)
        self.update_value_display()

# Attenuator Panel
class AttenuatorPanel(QWidget):
    def __init__(self, parent, name, device_serial):
        super().__init__(parent)
        self.name = name
        self.device_serial = device_serial
        self.step_size = config.ATTENUATOR_STEP_SIZES[0]
        self.value = 0  # Will be updated from device

        # Initialize the device
        self.device = Vaunix_LDA(name, device_serial, dll_path=config.DRIVERS_PATH, test_mode=False)
        self.device.working_frequency(config.CENTER_FREQ)

        # Read the current value from the device
        try:
            self.value = round(self.device.attenuation.get_raw(), 2)
        except Exception as e:
            print(f"Error reading initial value for {self.name}: {e}")
            self.value = 0  # Default value if read fails

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel(self.name)
        layout.addWidget(self.label)

        self.value_label = QLabel(f"Current Value: {self.value} dB")
        layout.addWidget(self.value_label)

        button_layout = QHBoxLayout()
        self.increase_button = QPushButton("↑")
        self.increase_button.clicked.connect(self.increase_value)
        button_layout.addWidget(self.increase_button)

        self.decrease_button = QPushButton("↓")
        self.decrease_button.clicked.connect(self.decrease_value)
        button_layout.addWidget(self.decrease_button)
        layout.addLayout(button_layout)

        # Entry to manually set value
        self.value_entry = QLineEdit()
        layout.addWidget(self.value_entry)

        # Set value button
        self.set_button = QPushButton("Set Value")
        self.set_button.clicked.connect(self.set_value)
        layout.addWidget(self.set_button)

        # Step size selection
        step_layout = QHBoxLayout()
        self.step_label = QLabel("Step Size:")
        step_layout.addWidget(self.step_label)

        self.step_combo = QComboBox()
        self.step_combo.addItems([str(s) for s in config.ATTENUATOR_STEP_SIZES])
        self.step_combo.currentIndexChanged.connect(self.update_step_size)
        step_layout.addWidget(self.step_combo)
        layout.addLayout(step_layout)

        self.setLayout(layout)

    def update_step_size(self):
        self.step_size = float(self.step_combo.currentText())

    def update_value_display(self):
        self.value_label.setText(f"Current Value: {self.value} dB")

    def increase_value(self):
        new_value = min(self.value + self.step_size, 50)
        self.set_device_value(new_value)

    def decrease_value(self):
        new_value = max(self.value - self.step_size, 0)
        self.set_device_value(new_value)

    def set_value(self):
        try:
            new_value = float(self.value_entry.text())
            if 0 <= new_value <= 50:
                self.set_device_value(new_value)
            else:
                QMessageBox.warning(self, "Invalid Input", "Value must be between 0 and 50 dB.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")

    def set_device_value(self, new_value):
        # Update the device value
        self.device.attenuation(new_value)
        self.value = round(new_value, 2)
        self.update_value_display()

# Phase Shifter Panel Container
class PhaseShifterPanelContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.panels = []

        layout = QVBoxLayout()

        # Use devices from config
        devices = config.PHASE_SHIFTER_DEVICES

        for device in devices:
            panel = PhaseShifterPanel(self, device["name"], device["serial"])
            layout.addWidget(panel)
            self.panels.append(panel)

        self.setLayout(layout)

# Attenuator Panel Container
class AttenuatorPanelContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.panels = []

        layout = QVBoxLayout()

        # Use devices from config
        devices = config.ATTENUATOR_DEVICES

        for device in devices:
            panel = AttenuatorPanel(self, device["name"], device["serial"])
            layout.addWidget(panel)
            self.panels.append(panel)

        self.setLayout(layout)

# VNA Control Panel
class VNAControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vna = None  # Will be initialized later

        # Baseline data
        self.baseline_power = None

        # Initialize selected frequency for vertical marker
        self.selected_freq = None

        # Previous parameters to detect changes
        self.prev_center_freq = None
        self.prev_span_freq = None
        self.prev_att = None
        self.prev_del_f = None

        self.is_updating = False  # Flag to prevent multiple threads

        # Add attribute to store current configuration
        self.current_configuration = "Normal Operation"
        self.num_peaks_to_find = 2  # Default to 2 peaks for Normal Operation

        self.initUI()

        # Start the timer for updating the plot
        self.timer = QTimer()
        self.timer.setInterval(1000)  # Update every 1000 ms (1 second)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # Control widgets
        control_layout = QGridLayout()
        self.center_freq_label = QLabel("Center Frequency (GHz):")
        self.center_freq_input = QLineEdit(str(config.VNA_DEFAULT_CENTER_FREQ / 1e9))  # Default value in GHz

        self.span_freq_label = QLabel("Span Frequency (MHz):")
        self.span_freq_input = QLineEdit(str(config.VNA_DEFAULT_SPAN_FREQ / 1e6))  # Default value in MHz

        self.att_label = QLabel("External Attenuation (dB):")
        self.att_input = QLineEdit(str(config.VNA_DEFAULT_ATTENUATION))  # Default value

        self.df_label = QLabel("Frequency Step ΔF (MHz):")
        self.df_input = QLineEdit(str(config.VNA_DEFAULT_DEL_F / 1e6))  # Default value in MHz

        self.store_thru_button = QPushButton("Store Thru")
        self.store_thru_button.clicked.connect(self.store_thru)

        control_layout.addWidget(self.center_freq_label, 0, 0)
        control_layout.addWidget(self.center_freq_input, 0, 1)
        control_layout.addWidget(self.span_freq_label, 1, 0)
        control_layout.addWidget(self.span_freq_input, 1, 1)
        control_layout.addWidget(self.att_label, 2, 0)
        control_layout.addWidget(self.att_input, 2, 1)
        control_layout.addWidget(self.df_label, 3, 0)
        control_layout.addWidget(self.df_input, 3, 1)
        control_layout.addWidget(self.store_thru_button, 4, 0, 1, 2)

        main_layout.addLayout(control_layout)

        # Matplotlib Figure and Canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(111)
        main_layout.addWidget(self.canvas)

        # Status bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)

        # Connect events for interactivity
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def update_plot(self):
        if not self.is_updating:
            self.is_updating = True
            threading.Thread(target=self.update_plot_thread).start()

    def update_plot_thread(self):
        # Get current settings
        try:
            center_freq = float(eval(self.center_freq_input.text())) * 1e9  # Convert GHz to Hz
            span_freq = float(eval(self.span_freq_input.text())) * 1e6  # Convert MHz to Hz
            att = float(eval(self.att_input.text()))
            del_f = float(eval(self.df_input.text())) * 1e6  # Convert MHz to Hz
        except Exception as e:
            print(f"Error parsing inputs: {e}")
            self.is_updating = False
            return

        # Check if VNA needs to be reinitialized
        needs_reinit = False
        if (self.prev_center_freq != center_freq or
            self.prev_span_freq != span_freq or
            self.prev_att != att or
            self.prev_del_f != del_f):
            needs_reinit = True
            self.prev_center_freq = center_freq
            self.prev_span_freq = span_freq
            self.prev_att = att
            self.prev_del_f = del_f

        # Initialize or update VNA settings
        if self.vna is None or needs_reinit:
            self.initialize_vna(center_freq, span_freq, att, del_f)
        else:
            self.vna.set_center_frequency(center_freq)
            self.vna.set_span_frequency(span_freq)
            self.vna.set_attenuation(att)
            self.vna.set_del_f(del_f)

        # Get new trace from VNA
        freqs, power_dbm, _ = self.vna.get_single_trace()

        # Subtract baseline if available
        if self.baseline_power is not None:
            power_dbm -= self.baseline_power

        # Update data
        self.freqs = freqs
        self.power_dbm = power_dbm

        # Now, update the plot in the main thread
        self.update_plot_canvas()

        self.is_updating = False

    def update_plot_canvas(self):
        # Clear axes
        self.ax1.clear()

        # Convert frequencies to GHz for plotting
        freqs_in_ghz = self.freqs / 1e9

        # Plot power_dbm
        self.ax1.plot(freqs_in_ghz, self.power_dbm, label='Trace')
        self.ax1.set_title("Power (dBm) vs Frequency")
        self.ax1.set_xlabel("Frequency (GHz)")
        self.ax1.set_ylabel("Power (dBm)")

        # Perform peak detection
        prominence = .1  # You can adjust this if needed
        peaks, properties = find_peaks(self.power_dbm, prominence=prominence)
        # Order peaks by prominence
        if len(peaks) > 0:
            prominences = properties["prominences"]
            sorted_indices = np.argsort(prominences)[::-1]  # Descending order
            # Determine the number of peaks to find based on the current configuration
            num_peaks = min(self.num_peaks_to_find, len(peaks))
            top_peaks = peaks[sorted_indices[:num_peaks]]
            # Plot the peaks
            for peak in top_peaks:
                freq = freqs_in_ghz[peak]
                power = self.power_dbm[peak]
                self.ax1.plot(freq, power, 'r*', markersize=10)
                # Annotate the peaks with frequency
                self.ax1.annotate(f"{freq:.6f} GHz", (freq, power), textcoords="offset points", xytext=(0,10), ha='center', color='red')

        # Plot vertical line if selected_freq is set
        if self.selected_freq is not None:
            self.ax1.axvline(x=self.selected_freq / 1e9, color='r', linestyle='--')

        self.ax1.legend()

        # Refresh canvas
        self.canvas.draw()

    def initialize_vna(self, center_freq, span_freq, att, del_f):
        print(f"Initializing VNA with center_freq={center_freq} Hz, span_freq={span_freq} Hz, att={att}, del_f={del_f} Hz")
        # Re-initialize VNA object
        self.vna = VectorNetworkAnalyzer(
            center_freq=center_freq,
            span_freq=span_freq,
            att_into_qm_box=att,
            lo_serial_number=config.VNA_LO_SERIAL_NUMBER,
            base_dir=config.VNA_BASE_DIR,
            n_avg=config.VNA_N_AVG,  # For faster updates
            bandwidth=span_freq,
            del_f=del_f  # Frequency step
        )

    def store_thru(self):
        # Store current trace as baseline
        print("Storing current trace as baseline")
        self.baseline_power = self.power_dbm.copy()

    def on_hover(self, event):
        # Display the x and y data in the status bar
        if event.inaxes == self.ax1:
            if event.xdata is not None and event.ydata is not None:
                freq_in_ghz = event.xdata
                self.status_bar.showMessage(f"Frequency: {freq_in_ghz:.6f} GHz, Power: {event.ydata:.2f} dBm")
            else:
                self.status_bar.clearMessage()
        else:
            self.status_bar.clearMessage()

    def on_click(self, event):
        if event.inaxes == self.ax1:
            if event.xdata is not None:
                self.selected_freq = event.xdata * 1e9  # Convert GHz back to Hz
                self.update_plot_canvas()

    def closeEvent(self, event):
        # Turn off VNA when closing the application
        if self.vna:
            self.vna.turn_off()
        event.accept()

    def on_configuration_changed(self, configuration_name):
        # Update current configuration
        self.current_configuration = configuration_name
        if configuration_name == "Normal Operation":
            self.num_peaks_to_find = 2
        else:
            self.num_peaks_to_find = 1

# Main Application
class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Combined GUI Application")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left layout for controls
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_scroll_area.setWidget(left_widget)
        main_layout.addWidget(left_scroll_area, stretch=1)

        # Right layout for VNA
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=2)

        # Add Voltage Control Panel
        self.voltage_control_panel = VoltageControlPanel()
        left_layout.addWidget(self.voltage_control_panel)

        # Add Switch Configuration Panel
        self.switch_config_panel = SwitchConfigurationPanel()
        left_layout.addWidget(self.switch_config_panel)

        # Add Phase Shifter Panels
        self.phase_shifter_panel = PhaseShifterPanelContainer()
        left_layout.addWidget(self.phase_shifter_panel)

        # Add Attenuator Panels
        self.attenuator_panel = AttenuatorPanelContainer()
        left_layout.addWidget(self.attenuator_panel)

        # Add a stretch to left layout to push components to the top
        left_layout.addStretch()

        # Add VNA Control Panel
        self.vna_control_panel = VNAControlPanel()
        right_layout.addWidget(self.vna_control_panel)

        # Connect configuration_changed signal to VNA control panel
        self.switch_config_panel.configuration_changed.connect(self.vna_control_panel.on_configuration_changed)

    def closeEvent(self, event):
        # Close devices properly
        self.vna_control_panel.closeEvent(event)
        super().closeEvent(event)

# Run the application
def main():
    app = QApplication(sys.argv)
    window = MainApplication()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
