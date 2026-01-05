import telnetlib3
import numpy as np
import re
import time

TIMEOUT = 3 # seconds to wait for responses before concluding something is wrong or moving on

class Digilock_UI:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        MAX_RETRIES = 5

        for attempt in range(MAX_RETRIES):
            print(f"Connecting to {self.host}:{self.port} (attempt {attempt+1}/{MAX_RETRIES})...")
            try:
                self.tn = telnetlib3.Telnet(host, port, timeout=5)
                resp = self.tn.read_until(b"> ", timeout=10)
                if resp.strip():
                    print("Connected successfully!")
                    break
            except Exception as e:
                print(f"Telnet connection failed: {e}")
            time.sleep(2)
        else:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port} after {MAX_RETRIES} attempts")
        
        #print(self.tn.read_until(b"> ", timeout=10).split(b'\r')[2].decode('ascii')) # wait to first prompt
        print('-----------------------------------------------')
        print()
        
        
    def close (self) -> None:
        self.tn.close()
        
    def query_lines(self, command: str) -> str: # functional block that separates good data from chaff
        self.tn.write((command+"?").encode('ascii')+b"\n")
        return_str = self.tn.read_until(b"> ", timeout=TIMEOUT).decode('ascii')
        if not return_str: 
            raise RuntimeError(command + ": NO RESPONSE")
        if 'Error' in return_str: # makes incorrect commands a little more transparent
            raise TypeError(f'[{command}] is unknown command')
        
        split_str = return_str.split('=')[-1] # get rid of all echo
        split_str = split_str.split('\r')[0] # get rid of trailing stuff
        return split_str

    def query_numeric(self, command: str) -> float:
        line = self.query_lines(command)
        if 'm' in line: # some of the DUIs use a.u. some return actual values with order of mag suffix
            mult = 1e-3
        elif 'u' in line:
            mult = 1e-6
        elif 'n' in line:
            mult = 1e-9
        else:
            mult = 1
        cleaned = re.sub(r'[^0-9+.\-]', '', line) # get rid of all non numeric chars
        return float(cleaned) * mult

    def query_range(self, command: str) -> str:
        line = self.query_lines(command+'.range')
        return line.strip()

    def query_enum(self, command: str) -> str:
        line = self.query_lines(command)
        return line.strip()
    
    def query_bool(self, command:  str) -> bool:
        line = self.query_lines(command)
        return line.strip().lower() == "true"

    def query_graph(self, command: str) -> str: # functional block that separates good data from chaff
        self.tn.write((command+"?").encode('ascii')+b"\n")
        return_str = self.tn.read_until(b"> ", timeout=TIMEOUT).decode('ascii')
        if not return_str: 
            raise RuntimeError(command + ": NO RESPONSE")
        if 'Error' in return_str: # makes incorrect commands a little more transparent
            raise TypeError(f'[{command}] is unknown command')
        data_str = return_str.split('=')[1]
        data_str = data_str.split('\r\n')[:-1]
        data_list = []
        for i in range(len(data_str)):
            data_list.append([float(s) for s in data_str[i].split('\t')])
        data_array = np.array(data_list)
        #t = data_array[:,0]
        ch1 = data_array[:,1] #skip 2 cause its a redundant time array
        ch2 = data_array[:,3]
        #return t, ch1, ch2
        return ch1, ch2

    def send_comand(self, command: str) -> None :
        self.tn.write(command.encode('ascii')+b"\n")
        return_str = self.tn.read_until(b"> ", timeout=TIMEOUT).decode('ascii')
        if not return_str:
            raise RuntimeError(command + ": NO RESPONSE")
        if "Error" in return_str:
            if "bad command" in return_str:
                err_msg = f"ERROR: [{command}] is unknown command"
            elif "bad parameter" in return_str:
                err_msg = f"ERROR: [{command}] has bad parameter"
            elif "value out of range" in return_str:
                err_msg = f"ERROR: [{command}] parameter is out of range"
            else:
                err_msg = f"ERROR: [{command}] is read only command"
            raise TypeError(err_msg)


    def set_numeric(self, command: str, value: float) -> None:
        self.send_comand(command+"={}".format(value))
    
    def set_bool(self, command: str, value: bool) -> None:
        if bool:
            tf='true'
        else:
            tf='false'
        self.send_comand(command+"="+tf)

    def set_enum(self, command: str, item: str) -> None:   
        self.send_comand(command+"="+item)
        