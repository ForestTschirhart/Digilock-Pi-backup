from digilock_remote import Digilock_UI
from digilock_remote import Command_type
import matplotlib.pyplot as plt

dui = Digilock_UI("192.168.10.3", 60002)

for command in dui.commandset:
    if command.settable:
        print(command.name+'.range='+dui.query_range(command.name))

for command in dui.commandset:
    if command.queryable:
        if command.type == Command_type.Enum:
            print('{command.name}={value}'.format(command=command, value=dui.query_enum(command)))
        if command.type == Command_type.Numeric:
            print('{command.name}={value}'.format(command=command, value=dui.query_numeric(command)))
        if command.type == Command_type.Bool:
            print('{command.name}={value}'.format(command=command, value=dui.query_bool(command)))

print()
cur_gain = dui.query_numeric('pid1:gain')
print(cur_gain)
dui.set_numeric('pid1:gain', 21)
cur_gain = dui.query_numeric('pid1:gain')
print(cur_gain)



dui.close()