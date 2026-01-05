from telnetlib3 import Telnet

tn = Telnet("192.168.10.3", 60001)

tn.read_until(b"> ", timeout=10)

tn.write("scope:ch1:mean?".encode('ascii')+b"\n")
return_str = tn.read_until(b"> ", timeout=1).decode('ascii')
print(return_str)
splitted_str = return_str.split('\n')
del splitted_str[-1]
del splitted_str[0]
print(splitted_str)
print(splitted_str[0].split('='))
splitted_str[0] = splitted_str[0].split('=')[1]
print(splitted_str)
for command in splitted_str:
    print(command)

tn.write("exit".encode('ascii')+b"\n")
tn.read_all()
tn.close()