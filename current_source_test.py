import sys 
if "S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers" not in sys.path:
    sys.path.append("S:\\fitzlab\\code\\QM_fitzlab\\instrument_drivers")

from CS580 import CS580


cs = CS580('cs', 'ASRL4::INSTR')

idn = cs.get_idn()

print(idn)

current_value = cs.current.get()

print(current_value)

cs.current.set(-1.501e-02)