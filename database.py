# database.py

from sqlalchemy import create_engine, Column, Integer, Float, String, text
from sqlalchemy.orm import sessionmaker, declarative_base
import os

Base = declarative_base()


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
