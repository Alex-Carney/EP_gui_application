import os
import csv
from sqlalchemy import create_engine, Column, Integer, Float, String, text
from sqlalchemy.orm import sessionmaker, declarative_base
import os

Base = declarative_base()


def aggregate_csvs_into_db(csv_folder, db_name='experiment_data.db'):
    """
    Scans `csv_folder` for all .csv files matching the format we saved in the VNA panel,
    and inserts them into the DB defined by EPMeasurementModel.
    """
    # Create or open the DB
    SessionFactory = create_database(db_name=db_name, folder_path='./databases')
    session = SessionFactory()

    # Scan for CSV files
    for fname in os.listdir(csv_folder):
        if not fname.lower().endswith('.csv'):
            continue

        full_path = os.path.join(csv_folder, fname)
        print(f"Processing {full_path} ...")

        with open(full_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # The DictReader will expect columns: 
            # "experiment_id", "readout_type", "set_voltage", "frequency_hz", "power_dBm"
            # If your CSV has more columns, adapt accordingly.

            # Insert row by row
            row_count = 0
            for row in reader:
                try:
                    experiment_id = row.get("experiment_id", "manual_experiment")
                    readout_type = row.get("readout_type", "unknown")
                    set_voltage = float(row.get("set_voltage", 0.0))
                    frequency_hz = float(row.get("frequency_hz", 0.0))
                    power_dBm = float(row.get("power_dBm", 0.0))

                    # Build the EPMeasurementModel object
                    measurement = EPMeasurementModel(
                        experiment_id=experiment_id,
                        readout_type=readout_type,
                        set_voltage=set_voltage,
                        frequency_hz=frequency_hz,
                        power_dBm=power_dBm
                    )
                    # If you want to store more columns, e.g. loop_phase, etc., parse them from row too.

                    session.add(measurement)
                    row_count += 1
                except ValueError as ve:
                    print(f"Skipping row in {fname} due to parse error: {ve}")
                except Exception as e:
                    print(f"Unknown error in {fname}: {e}")

            print(f"Inserted {row_count} rows from {fname} into the DB.")

    session.commit()
    session.close()
    print("All CSV files have been aggregated into the database.")


class EPMeasurementModel(Base):
    __tablename__ = 'expr'
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, index=True)
    # Measurement
    frequency_hz = Column(Float)
    power_dBm = Column(Float)

    # Readout type
    readout_type = Column(String)  # 'normal', 'cavity', 'yig'

    # Additional variables
    omega_C = Column(Float)
    omega_Y = Column(Float)
    kappa_C = Column(Float)
    kappa_Y = Column(Float)
    Delta = Column(Float)
    K = Column(Float)

    # Loop
    set_loop_phase_deg = Column(Float)
    set_loop_att = Column(Float)
    set_loopback_att = Column(Float)

    # YIG feedback
    set_yig_fb_phase_deg = Column(Float)
    set_yig_fb_att = Column(Float)

    # Cavity feedback
    set_cavity_fb_phase_deg = Column(Float)
    set_cavity_fb_att = Column(Float)

    # Independent variable
    set_voltage = Column(Float)


def create_database(db_name='experiment_data.db', folder_path='./databases'):
    # Ensure the database file name ends with .db
    if not db_name.endswith('.db'):
        db_name += '.db'

    # Ensure the folder path exists
    os.makedirs(folder_path, exist_ok=True)

    # Combine the folder path and database name
    db_path = os.path.join(folder_path, db_name)

    # Create the database engine with the full path
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)

    # Enable WAL mode
    with engine.connect() as connection:
        connection.execute(text("PRAGMA journal_mode=WAL;"))

    return sessionmaker(bind=engine)


if __name__ == "__main__":
    # Example usage
    csv_folder_path = r"C:\Users\AlexPHD\Dartmouth College Dropbox\Alexander Carney\darpa_alex_local\EP_gui_application\manual_sweeps\second_sweep_30_dB_good_regime"

    aggregate_csvs_into_db(csv_folder_path, db_name='THE_SECOND_MANUAL.db')
