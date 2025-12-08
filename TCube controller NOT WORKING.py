from thorlabs_apt_device import TDC001
from serial.tools.list_ports import comports

serial_ports = [(x[0], x[1], dict(y.split('=', 1) for y in x[2].split(' ') if '=' in y)) for x in comports()]

print(serial_ports)

from serial.tools.list_ports_windows import *

def GetList(self, verbose=True):
        """
            gets the list of all available ports
        """

        results = []

        hits = 0

        iterator = sorted(comports())

        # list them
        for port, desc, hwid in iterator:
            comPort = port
            if verbose:
                descValue = desc
                hwidValue = hwid
                results.append({'comPort': comPort, 'descValue': descValue, 'hwidValue': hwidValue})
            else:
                results.append({'comPort': comPort})
            hits += 1

        results.append({"available": "{} ports found".format(hits)})

        return results

import wmi
c = wmi.WMI()
wql = "Select * From Win32_SerialPort"
for item in c.query(wql):
    print('hello')
    print (item)

import serial.tools.list_ports
print(list(serial.tools.list_ports.comports()))
# servo=TDC001()
