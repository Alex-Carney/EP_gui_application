#!/usr/bin/env python
"""
Provides support for the SRS DC 205 power supply.
"""

# IMPORTS #####################################################################


from enum import IntEnum

from instruments.units import ureg as u

from instruments.abstract_instruments import PowerSupply
from instruments.abstract_instruments import Instrument
from instruments.util_fns import assume_units, ProxyList

# CLASSES #####################################################################


class SRSDC205(PowerSupply, Instrument):

    """
    The SRS DC 205 is a single channel DC power supply.
    Example usage:
    >>> import instruments as ik
    >>> import instruments.units as u
    >>> inst = ik.SRS.SRSDC205.open_gpibusb("/dev/ttyUSB0", 1)
    >>> inst.voltage = 10 * u.V
    """

    # INNER CLASSES #

    class Channel(PowerSupply.Channel):

        """
        Class representing the only channel on the SRS DC 205.
        This class inherits from `PowerSupply.Channel`.
        .. warning:: This class should NOT be manually created by the user. It
            is designed to be initialized by the `SRSDC205` class.
        """

        def __init__(self, parent, name):
            self._parent = parent
            self._name = name

        # PROPERTIES #

        @property
        def mode(self):
            """
            Sets the output mode for the power supply channel.
            This is either constant voltage.
            Querying the mode is not supported by this instrument.
            :type: `SRSDC205.Mode`
            """
            raise NotImplementedError(
                "This instrument does not support " "querying the operation mode."
            )

        @mode.setter
        def mode(self, newval):
            if not isinstance(newval, SRSDC205.Mode):
                raise TypeError(
                    "Mode setting must be a `SRSDC205.Mode` "
                    "value, got {} instead.".format(type(newval))
                )
            self._parent.sendcmd(f"F{newval.value};")
            self._parent.trigger()

        @property
        def voltage(self):
            """
            Sets the voltage of the specified channel. This device has a voltage
            range of 0V to +10V.
            Querying the voltage is not supported by this instrument.
            :units: As specified (if a `~pint.Quantity`) or
                assumed to be of units Volts.
            :type: `~pint.Quantity` with units Volt
            """
            raise NotImplementedError(
                "This instrument does not support "
                "querying the output voltage setting."
            )

        @voltage.setter
        def voltage(self, newval):
            newval = assume_units(newval, u.volt).to(u.volt).magnitude
            self.mode = self._parent.Mode.voltage
            self._parent.sendcmd(f"VOLT {newval};")
            self._parent.trigger()

        @property
        def current(self):
            """
            Sets the current of the specified channel. This device has an max
            setting of 100mA.

            Querying the current is not supported by this instrument.

            :units: As specified (if a `~pint.Quantity`) or
                assumed to be of units Amps.
            :type: `~pint.Quantity` with units Amp
            """
            raise NotImplementedError(
                "This instrument does not support "
                "querying the output current setting."
            )

        @current.setter
        def current(self, newval):
            newval = assume_units(newval, u.amp).to(u.amp).magnitude
            self.mode = self._parent.Mode.voltage
            self._parent.sendcmd(f"SA{newval};")
            self._parent.trigger()

        @property
        def output(self):
            """
            Sets the output status of the specified channel. This either enables
            or disables the output.
            Querying the output status is not supported by this instrument.
            :type: `bool`
            """
            raise NotImplementedError(
                "This instrument does not support " "querying the output status."
            )

        @output.setter
        def output(self, newval):
            if newval is True:
                self._parent.sendcmd("SOUT {newval};")
                self._parent.trigger()
            else:
                pass
                self._parent.sendcmd("O0;")
                self._parent.trigger()

    # ENUMS #

    class Mode(IntEnum):
        """
        Enum containing valid output modes for the SRS DC 205
        """

        voltage = 1

    # PROPERTIES #

    @property
    def channel(self):
        """
        Gets the specific power supply channel object. Since the SRSDC205
        is only equiped with a single channel, a list with a single element
        will be returned.
        This (single) channel is accessed as a list in the following manner::
        >>> import instruments as ik
        >>> yoko = ik.SRS.SRSDC205.open_gpibusb('/dev/ttyUSB0', 10)
        >>> yoko.channel[0].voltage = 1 # Sets output voltage to 1V
        :rtype: `~SRSDC205.Channel`
        """
        return ProxyList(self, SRSDC205.Channel, [0])

    @property
    def voltage(self):
        """
        Sets the voltage. This device has a voltage range of 0V to +30V.
        Querying the voltage is not supported by this instrument.
        :units: As specified (if a `~pint.Quantity`) or assumed
            to be of units Volts.
        :type: `~pint.Quantity` with units Volt
        """
        raise NotImplementedError(
            "This instrument does not support querying " "the output voltage setting."
        )

    @voltage.setter
    def voltage(self, newval):
        self.channel[0].voltage = newval

    @property
    def current(self):
        """
        Sets the current. This device has an max setting of 100mA.

        Querying the current is not supported by this instrument.

        :units: As specified (if a `~pint.Quantity`) or assumed
            to be of units Amps.
        :type: `~pint.Quantity` with units Amp
        """
        raise NotImplementedError(
            "This instrument does not support querying " "the output current setting."
        )

    @current.setter
    def current(self, newval):
        self.channel[0].voltage = newval

    # METHODS #

    def trigger(self):
        """
        Triggering function for the Yokogawa 7651.

        After changing any parameters of the instrument (for example, output
        voltage), the device needs to be triggered before it will update.
        """
        self.sendcmd("E;")

