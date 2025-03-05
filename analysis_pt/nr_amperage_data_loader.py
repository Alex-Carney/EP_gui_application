# data_access.py
import pandas as pd
from sqlalchemy import create_engine
from stages import DataLoader


class NRAmperageDataLoader(DataLoader):
    def __init__(self, db_path, experiment_id, readout_type,
                 freq_min, freq_max, current_min, current_max, independent_var="set_amperage"):
        self.db_path = db_path
        self.experiment_id = experiment_id
        self.readout_type = readout_type
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.current_min = current_min
        self.current_max = current_max
        self.independent_var = independent_var
        self.engine = create_engine(f"sqlite:///{db_path}")

    def load_data(self):
        settings_query = f"""
        SELECT DISTINCT set_loop_att, set_loopback_att,
                        set_yig_fb_phase_deg, set_yig_fb_att,
                        set_cavity_fb_phase_deg, set_cavity_fb_att
        FROM expr
        WHERE experiment_id = '{self.experiment_id}' AND readout_type = '{self.readout_type}'
        """
        settings_df = pd.read_sql_query(settings_query, self.engine)
        settings = settings_df.iloc[0].to_dict() if not settings_df.empty else {}

        data_query = f"""
        SELECT frequency_hz, {self.independent_var} as current, power_dBm
        FROM expr
        WHERE experiment_id = '{self.experiment_id}'
          AND readout_type = '{self.readout_type}'
          AND {self.independent_var} BETWEEN {self.current_min} AND {self.current_max}
          AND frequency_hz BETWEEN {self.freq_min} AND {self.freq_max}
        ORDER BY current, frequency_hz
        """
        data = pd.read_sql_query(data_query, self.engine)
        if data.empty:
            return None, None, None, None
        pivot_table = data.pivot_table(index="current", columns="frequency_hz",
                                       values="power_dBm", aggfunc="first")
        currents = pivot_table.index.values
        frequencies = pivot_table.columns.values
        power_grid = pivot_table.values
        return power_grid, currents, frequencies, settings
