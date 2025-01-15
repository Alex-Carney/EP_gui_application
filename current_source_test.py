import sys

import config
if config.DRIVERS_PATH not in sys.path:
    sys.path.append(config.DRIVERS_PATH)

from CS580 import CS580


cs = CS580('cs', 'ASRL4::INSTR')

idn = cs.get_idn()

print(idn)

current_value = cs.current.get()

print(current_value)

cs.current.set(-1.501e-02)