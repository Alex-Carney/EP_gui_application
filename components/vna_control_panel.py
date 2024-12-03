# components/vna_control_panel.py

import threading
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QStatusBar, QMessageBox, QFileDialog
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks, peak_widths
import config
from vna.vna_instrument import VectorNetworkAnalyzer

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

        # Initialize omega_Y, kappa_Y, omega_C, kappa_C
        self.omega_Y = None
        self.kappa_Y = None
        self.omega_C = None
        self.kappa_C = None

        # Add a lock to protect VNA access
        self.vna_lock = threading.Lock()

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

        # Add Save Data button
        self.save_data_button = QPushButton("Save Data")
        self.save_data_button.clicked.connect(self.save_data)
        main_layout.addWidget(self.save_data_button)

        # Initialize information labels
        self.init_info_labels(main_layout)

        # Status bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)

        # Connect events for interactivity
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def init_info_labels(self, main_layout):
        # Create a horizontal layout
        info_layout = QHBoxLayout()
        main_layout.addLayout(info_layout)

        # Column 1: YIG information
        self.yig_info_layout = QVBoxLayout()
        self.yig_info_label = QLabel("YIG Information")
        self.omega_y_label = QLabel("ω_Y = N/A")
        self.kappa_y_label = QLabel("κ_Y = N/A")
        self.yig_info_layout.addWidget(self.yig_info_label)
        self.yig_info_layout.addWidget(self.omega_y_label)
        self.yig_info_layout.addWidget(self.kappa_y_label)
        info_layout.addLayout(self.yig_info_layout)

        # Column 2: Cavity information
        self.cavity_info_layout = QVBoxLayout()
        self.cavity_info_label = QLabel("Cavity Information")
        self.omega_c_label = QLabel("ω_C = N/A")
        self.kappa_c_label = QLabel("κ_C = N/A")
        self.cavity_info_layout.addWidget(self.cavity_info_label)
        self.cavity_info_layout.addWidget(self.omega_c_label)
        self.cavity_info_layout.addWidget(self.kappa_c_label)
        info_layout.addLayout(self.cavity_info_layout)

        # Column 3: Delta and K
        self.delta_info_layout = QVBoxLayout()
        self.delta_info_label = QLabel("Delta and K")
        self.delta_label = QLabel("Δ = N/A")
        self.K_label = QLabel("K = N/A")
        self.delta_info_layout.addWidget(self.delta_info_label)
        self.delta_info_layout.addWidget(self.delta_label)
        self.delta_info_layout.addWidget(self.K_label)
        info_layout.addLayout(self.delta_info_layout)

    def update_plot(self):
        if not self.is_updating:
            self.is_updating = True
            threading.Thread(target=self.update_plot_thread).start()

    def update_plot_thread(self):
        with self.vna_lock:
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

            # Update VNA settings
            if self.vna is None:
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
        prominence = 0.1  # Adjust as needed
        peaks, properties = find_peaks(self.power_dbm, prominence=prominence)
        # Order peaks by prominence
        if len(peaks) > 0:
            prominences = properties["prominences"]
            sorted_indices = np.argsort(prominences)[::-1]  # Descending order
            # Determine the number of peaks to find based on the current configuration
            num_peaks = min(self.num_peaks_to_find, len(peaks))
            top_peaks = peaks[sorted_indices[:num_peaks]]
            # Compute FWHM
            results_half = peak_widths(self.power_dbm, top_peaks, rel_height=0.5)
            widths = results_half[0]
            width_heights = results_half[1]
            left_ips = results_half[2]
            right_ips = results_half[3]
            # Plot the peaks and FWHM
            for i, peak in enumerate(top_peaks):
                freq = freqs_in_ghz[peak]
                power = self.power_dbm[peak]
                self.ax1.plot(freq, power, 'r*', markersize=10)
                # Annotate the peaks with frequency
                self.ax1.annotate(f"{freq:.6f} GHz", (freq, power), textcoords="offset points", xytext=(0,10), ha='center', color='red')

                # Draw horizontal line at FWHM level
                width = widths[i]
                left_ip = left_ips[i]
                right_ip = right_ips[i]
                left_freq = freqs_in_ghz[int(left_ip)]
                right_freq = freqs_in_ghz[int(right_ip)]
                height = width_heights[i]
                self.ax1.hlines(height, left_freq, right_freq, color='green', linestyle='--')

                # Annotate FWHM
                fwhm = right_freq - left_freq  # In GHz
                self.ax1.annotate(f"FWHM: {fwhm*1e3:.3f} MHz", ((left_freq + right_freq)/2, height), textcoords="offset points", xytext=(0,-15), ha='center', color='green')

                # Update omega_Y, kappa_Y, omega_C, kappa_C
                if self.current_configuration == "YIG Readout Only":
                    self.omega_Y = freq
                    self.kappa_Y = fwhm
                    self.omega_y_label.setText(f"ω_Y = {freq:.6f} GHz")
                    self.kappa_y_label.setText(f"κ_Y = {fwhm*1e3:.3f} MHz")
                elif self.current_configuration == "Cavity Readout Only":
                    self.omega_C = freq
                    self.kappa_C = fwhm
                    self.omega_c_label.setText(f"ω_C = {freq:.6f} GHz")
                    self.kappa_c_label.setText(f"κ_C = {fwhm*1e3:.3f} MHz")

            # Update Delta and K
            if self.omega_Y is not None and self.omega_C is not None:
                delta = self.omega_C - self.omega_Y
                K = self.kappa_C - self.kappa_Y
                self.delta_label.setText(f"Δ = {delta*1e3:.3f} MHz")
                self.K_label.setText(f"K = {K*1e3:.3f} MHz")
            else:
                self.delta_label.setText("Δ = N/A")
                self.K_label.setText("K = N/A")

        # Plot vertical line if selected_freq is set
        if self.selected_freq is not None:
            self.ax1.axvline(x=self.selected_freq / 1e9, color='r', linestyle='--')

        self.ax1.legend()

        # Refresh canvas
        self.canvas.draw()

    def initialize_vna(self, center_freq, span_freq, att, del_f):
        print(f"Initializing VNA with center_freq={center_freq} Hz, span_freq={span_freq} Hz, att={att}, del_f={del_f} Hz")
        # Initialize VNA object
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
            self.vna.close()
        event.accept()

    def on_configuration_changed(self, configuration_name):
        # Update current configuration
        self.current_configuration = configuration_name
        if configuration_name == "Normal Operation":
            self.num_peaks_to_find = 2
        else:
            self.num_peaks_to_find = 1

    def save_data(self):
        if not hasattr(self, 'freqs') or not hasattr(self, 'power_dbm'):
            QMessageBox.warning(self, "No Data", "No data to save.")
            return

        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Data", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if filename:
            import csv
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Frequency (Hz)', 'Power (dBm)'])
                for f, p in zip(self.freqs, self.power_dbm):
                    csvwriter.writerow([f, p])
            print(f"Data saved to {filename}")


    def collect_data_sync(self):
        # Collect data synchronously
        # No need to lock here; the calling method should handle locking
        try:
            center_freq = float(eval(self.center_freq_input.text())) * 1e9  # Convert GHz to Hz
            span_freq = float(eval(self.span_freq_input.text())) * 1e6  # Convert MHz to Hz
            att = float(eval(self.att_input.text()))
            del_f = float(eval(self.df_input.text())) * 1e6  # Convert MHz to Hz
        except Exception as e:
            print(f"Error parsing inputs: {e}")
            self.is_updating = False
            return

        # Update VNA settings
        if self.vna is None:
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

    def get_current_trace(self):
        with self.vna_lock:
            # Collect data synchronously
            self.collect_data_sync()
            return self.freqs, self.power_dbm
