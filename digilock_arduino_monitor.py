import serial
from digilock_remote import Digilock_UI
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import ORJSONResponse, JSONResponse
import threading
import numpy as np
import matplotlib.pyplot as plt
import time
import RPi.GPIO as GPIO
import requests

GPIO.setwarnings(False)  # Suppresses the warning
GPIO.setmode(GPIO.BOARD)

"""
Two threads monitor the two lasers through through the digilockRCI for signs of unlocking or if they are fully unlocked.
Another thread monitors the arduino pin logic flags.
Each thread modifies global variables stored in the GLOB_DICT in case the threads need to share blocking info later for feedback

When any of the monitor loops senses trouble they PUSH to the dash server, making sure that the logging functionality catches changes of state

Additionally, the two rci monitors and the arduino can be queried for scope traces and have updated feedback parameters pushed to them

"""

### vvv just a list of relevant commands from the digilock rci interface module vvv ###
"""
needs2werk = ['scope:ch2:rms', 'pid2:lock:state', 'pid1:input', 'pid1:output', 'pid1:lock:enable', 'pid1:setpoint', 'pid1:sign', 'pid1:slope']
types = [           'num',     		'bool', 		'enum (str)',  'enum (str)',       'bool',             'num',		'bool',   	'bool']
able =  [            'q',            'q',                'q,s',      'q,s',             'q,s',           'q,s', 		'q,s', 		'q,s']
wouldBcool = ['scope:graph']
"""

# defualt init vars for the rms monitors

RMS_THRESH_G = 10
RMS_THRESH_B = 80
WINDOW_LEN_G = 20
WINDOW_LEN_B = 20
F_SAMP = 2 #Hz for the RMS loops

# GPIO State pins for the arduino connection
MOD_FAILURE_PIN = 29
MOD_ACTIVE_PIN = 31
MOD_EN_PIN = 33
LOCK_STATE_PIN = 35
PEAKS_LOST_PIN = 38 


N = 500 # arduino trace length


glob_dict_lock = threading.Lock() # for different classes modifying the same dict or 
tcp_lock = threading.Lock() # for interfacing with the digilock RCI over using tcp
push_2dash_lock = threading.Lock() # for talking to the dash from inside a thread

# Will need this in the future if we want multiple possible feedback sources (rms based vs. arduino)
GLOB_DICT = {'blue_happy': None,
             'blue_locked': None,
             'blue_rms_fb_active': None,
             'green_happy': None,
             'green_locked': None,
             'green_rms_fb_active': None,
             'ard_peaks': None,
             'ard_fb_active' : None,
             'ard_fb_failure': None,
             }


DASH_URL = "http://10.155.94.105:8050" # for pushing to the dash

def push_to_dash():
    try:
        r = requests.get(f"{DASH_URL}/api/digi_feedback_status", timeout=2)
        if r.ok:
            status1 = r.json().get("test1", False)
            status2 = r.json().get("test2", False)
        print(f'pushed to dash: {status1} {status2}')
    except Exception as e:
        print(f'frik >:(  {e}' )


class DUIMonitor:
    def __init__(self, name, ip, port, f_samp, window_len, rms_thresh, fill_frac_thresh, trace_downsamp_factor=10):
        self.name = name
        print(name)
        self.dui = Digilock_UI(ip, port)
        print()
        
        self.locked = False 
        self.lock_unhappy = False
        self.prev_state = [self.locked, self.lock_unhappy]
        
        self.std = None
        self.mean = None
        
        self.window_len = window_len
        self.rms_thresh = rms_thresh
        self.sum_thresh = int(window_len*fill_frac_thresh)
        self.running_window = np.zeros(window_len, dtype=bool) # for dynamic size edit outside of here
        
        self.f_samp = f_samp
        self.cur_ctrl_en = False
        self.cur_ctrl_active = False
        
        self.downsamp = int(trace_downsamp_factor)
        
        self.thread_lock = threading.Lock() # for modifying class attributes non-atomically from outside/inside the monitor loop thread
        
    def simple_monitor_loop(self):
        while True:
            try:              
                ti=time.perf_counter()
                with tcp_lock:
                    self.rms = self.dui.query_numeric('scope:ch2:rms')
                    self.mean = self.dui.query_numeric('scope:ch1:mean')
                    self.locked = self.dui.query_bool('pid2:lock:state')
                
                with self.thread_lock:
                    self.running_window[1:] = self.running_window[:-1] #bitshift
                    self.running_window[0] = self.rms > self.rms_thresh # add new bit to front of sliding window 
                    self.lock_unhappy = np.sum(self.running_window) >= self.sum_thresh
                
                # just for rms controlled feedback, now maybe defunct
                if self.lock_unhappy and self.locked and self.cur_ctrl_en:  # set up for current bump 
                    self.trigger_current_bump()
                
                # for state change push (logging)
                new_state = [self.locked, self.lock_unhappy]
                if (new_state != self.prev_state):
                    self.prev_state = new_state
                    push_to_dash(new_state) # to be modded later
                                
                tf=time.perf_counter()
                wait = (1/self.f_samp) - (tf-ti)
                if wait > 0:
                    time.sleep(wait)
                else:
                    print(f"LOOP LAGGING DESIRED F_SAMP by: {-1*wait} ms")
        
            except Exception as e:
                print(f"[{self.name} monitor error] {e}")
                time.sleep(2)
                
    
    def trigger_current_bump(self):
        # self.locked = self.dui.query_bool('pid2:lock:state') MAKE SURE WERE LOCKED BEFORE TRYING THIS CURRENT BUMP
        # safe loop inc = 0 
        #while safe loop inc < #:
            #bump current by (X)
            #loop to check if rms decreasing
                # roll and update running window
                # wait
            # sum window to see if below thresh
            # if not:
                #if we're still within output range
                    # set (X=bump inc) (loop and bump)
                # if not
                    # return current bump failed (deal with how to handle this in the main loop)       
            # if yes:
                # set current bump (X=0) (loop but dont bump next time)
                # set safe_loop_inc += 1
        print('current ctrl on')
        self.cur_ctrl_active = True
    
        
    def refresh_params(self, rms_thresh: float, window_len: int, fill_thresh: int):
        try:
            with self.thread_lock:
                self.rms_thresh = rms_thresh
                self.window_len = window_len
                self.sum_thresh = fill_thresh
                self.running_window = np.zeros(window_len, dtype=bool) # this is the one  that needs a lock (not atomic)
            #return [self.rms_thresh, self.window_len, fill_frac_thresh]
        except Exception as e:
            raise RuntimeError('Failed to Set Params') from e
    
    def get_traces(self):
        try:
            with tcp_lock:
                ch1, ch2 = self.dui.query_graph('scope:graph')   
            ch1, ch2 = ch1[::self.downsamp].tolist(), ch2[::self.downsamp].tolist()
            return ch1, ch2
        except Exception as e:
            print(f'{self.name} digilock trace fetch error: {e}')
            raise RuntimeError(f'{self.name} digilock trace fetch error') from e
          
        
class ArduinoMonitor:
    def __init__(self, serial_port, baud,
                 push_fbk_en_pin, push_digilock_status_pin,
                 pull_fbk_active_pin, pull_fbk_failure_pin, pull_peaks_lost_pin):
        
        try: 
            self.ser = serial.Serial(serial_port, baud, timeout=2)
            print("Arduino Connected!")
            time.sleep(0.5)  # Let serial port establish
            self.ser.reset_input_buffer()  # Clear arduino first time loop initialization message
                
        except Exception as e:
            print("Failed to open arduino serial port:", e)  
        
        self.thread_lock = threading.Lock()
        
        
        self.push_fbk_en_pin = push_fbk_en_pin
        self.push_digilock_status_pin = push_digilock_status_pin
        self.pull_fbk_active_pin = pull_fbk_active_pin
        self.pull_fbk_failure_pin = pull_fbk_failure_pin
        self.pull_peaks_lost_pin = pull_peaks_lost_pin
        GPIO.setup([push_fbk_en_pin, push_digilock_status_pin], GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup([pull_fbk_active_pin, pull_fbk_failure_pin, pull_peaks_lost_pin], GPIO.IN)
        
        # get params from arduino when pi script boots
        
        self.trigger_delay = None
        self.samp_ct = None
        
        self.query_params() # initializes params in group just above^
        
    def push_flag(self, pin, hilo):
        GPIO.output(pin, hilo)
        
    def pull_flag(self, pin):
        return GPIO.input(pin) == GPIO.HIGH
    
    def get_trace(self, N=500):
        self.ser.write(b'T\n')
        self.ser.flush()
        
        time.sleep(0.01)
    
    # Read EVERYTHING currently in buffer
        total_bytes = self.ser.in_waiting
        print(f"Arduino sent {total_bytes} bytes")
        
        raw = self.ser.read(N*2)
        
        return np.frombuffer(raw, dtype='<u2').astype(np.int32)
    
    def query_params(self):
        try:
            with self.thread_lock:
                self.ser.write(b'P\n')
                resp = self.ser.readline()
                resp = resp.decode('ascii').rstrip().split(",")
                self.trigger_delay = resp[0]
                self.samp_ct = resp[1]
        except Exception as e:
            print('Failed to query arduino params')
            raise RuntimeError('Failed to query arduino params') from e
    
    def refresh_params(self, trigger_delay: float, samp_ct: int):
        try:
            with self.thread_lock:
                self.ser.write(f'D{trigger_delay}\n'.encode('ascii'))
                resp = self.ser.readline()
                print(resp)
                self.trigger_delay = trigger_delay
                
                
                self.ser.write(f'C{samp_ct}\n'.encode('ascii'))
                self.samp_ct = samp_ct
                print(samp_ct)
                
        except Exception as e:
            print('Failed to Set Arduino Params')
            raise RuntimeError('Failed to Set Arduino Params') from e

        
app = FastAPI()

dui_blue = None # janky global variable fix for startup_event function variable scope issue
dui_green = None
ard_mon = None

@app.on_event("startup")
def startup_event():
    global dui_blue, dui_green, ard_mon
    # Create the DUIMonitor instances once at startup
    dui_blue = DUIMonitor('Blue', "192.168.10.3", 60001, F_SAMP, WINDOW_LEN_B, RMS_THRESH_B, fill_frac_thresh=0.75)
    dui_green = DUIMonitor('Green', "192.168.10.3", 60002, F_SAMP, WINDOW_LEN_G, RMS_THRESH_G, fill_frac_thresh=0.75)
    ard_mon = ArduinoMonitor('/dev/ttyACM0', 250000,
                             MOD_EN_PIN, LOCK_STATE_PIN,
                             MOD_ACTIVE_PIN, MOD_FAILURE_PIN, PEAKS_LOST_PIN) 
    
    # Start their background monitor threads
    threading.Thread(target=dui_blue.simple_monitor_loop, daemon=True).start()
    threading.Thread(target=dui_green.simple_monitor_loop, daemon=True).start()
    
    time.sleep(5)
    push_to_dash()

@app.post('/set_cur_ctrl')
def set_cur_ctrl(setting: dict = Body(...)):
    try:
        if setting['laser'] == 'blue':
            dui_blue.cur_ctrl_en = setting['value']
        elif setting['laser'] == 'green':
            dui_green.cur_ctrl_en = setting['value']
        else:
            raise ValueError('laser name invalid')
    except Exception as e:
        return HTTPException(status_code=500, detail=f'Current ctrl set failed: {e}')


@app.get("/digi_scope_traces", response_class=ORJSONResponse)
def get_scopes():
    try:
        b_ch1, b_ch2 = dui_blue.get_traces()
        g_ch1, g_ch2 = dui_green.get_traces()
        return {
            "b_ch1": b_ch1,
            "b_ch2": b_ch2,
            "g_ch1": g_ch1,
            'g_ch2': g_ch2
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scope Retrieval Failed: {e}")


@app.get("/arduino_trace", response_class=ORJSONResponse)
def get_arduino_trace():
    try:
        #with dui_blue.thread_lock:
        trace = ard_mon.get_trace() # wrap later
        
        return {
            "trace": trace.tolist(),
            "meta" : 0
        }
    except Exception as e:
        print(f"Arduino Trace Retrieval Failed: {e}")
        raise HTTPException(status_code=500, detail=f"Arduino Trace Retrieval Failed: {e}")



@app.get("/monitor_states", response_class=ORJSONResponse)
def get_lock_states():
    try:
        return {
            "b_locked": bool(dui_blue.locked),
            "b_lock_unhappy": bool(dui_blue.lock_unhappy),
            "g_locked": bool(dui_green.locked),
            "g_lock_unhappy": bool(dui_green.lock_unhappy),
            "b_cur_ctrl_active": bool(dui_blue.cur_ctrl_active),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monitor State Retrieval Failed: {e}")


@app.get('/monitor_params')
def get_monitor_params(laser: str = Query(...)):
    try:
        if laser == 'green':
            device = dui_green
        elif laser=='blue':
            device = dui_blue
        else:
            raise ValueError('laser name invalid')
        rms_thresh = device.rms_thresh
        window_len = device.window_len
        sum_thresh = device.sum_thresh
        return {
            "rms threshold": rms_thresh,
            "window length": window_len,
            "sum threshold": sum_thresh
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Monitor param retrieval failed: {e}')

@app.get('/arduino_params')
def get_arduino_params():
    try:
        ard_mon.query_params()
        trigger_delay = ard_mon.trigger_delay
        samp_ct = ard_mon.samp_ct
        return {
            "trigger delay": trigger_delay,
            "samp count" : samp_ct,
        }
    except Exception as e:
        print(f'Arduino param retrieval failed: {e}')
        raise HTTPException(status_code=500, detail=f'Arduino param retrieval failed: {e}')


@app.post("/refresh_params")
def set_lock_params(params: dict = Body(...)):
    ''' params format: { name: str(green / blue)
                         window length: int
                         rms threshold: float
                         fill fraction threshold: float
                        }
    '''
    
    try:
        device = dui_green if params['name'] == "green" else dui_blue
        device.refresh_params(params['rms threshold'],
                              params['window length'],
                              params['fill fraction threshold'])
        
        #return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parameter Refresh Failed: {e}")
   

@app.post("/refresh_arduino_params")
def set_ard_params(params: dict = Body(...)):
    ''' params format: { name: str(green / blue)
                         window length: int
                         rms threshold: float
                         fill fraction threshold: float
                        }
    '''
    
    try:
        ard_mon.refresh_params(params['trigger delay'],
                               params['samp count']
                               )
        #return {"status": "ok"}
    except Exception as e:
        print(f"Arduino Parameter Refresh Failed: {e}")
        raise HTTPException(status_code=500, detail=f"Arduino Parameter Refresh Failed: {e}")


if __name__ == '__main__':
    
    dui_blue = DUIMonitor('Blue', "192.168.10.3", 60001, F_SAMP, WINDOW_LEN_B, RMS_THRESH_B, fill_frac_thresh=0.75)
    dui_green = DUIMonitor('Green', "192.168.10.3", 60002, F_SAMP, WINDOW_LEN_G, RMS_THRESH_G, fill_frac_thresh=0.75)
    ard_mon = ArduinoMonitor('/dev/ttyACM0', 250000,
                             MOD_EN_PIN, LOCK_STATE_PIN,
                             MOD_ACTIVE_PIN, MOD_FAILURE_PIN, PEAKS_LOST_PIN) 
    
    print(dui_blue.dui.query_numeric('scope:ch2:rms'))
    print(dui_green.dui.query_numeric('scope:ch2:rms'))
    
    thread_blue = threading.Thread(target=dui_blue.simple_monitor_loop, daemon=True)
    thread_green = threading.Thread(target=dui_green.simple_monitor_loop, daemon=True)

    thread_green.start()
    thread_blue.start()
    
    trace = ard_mon.get_trace()
    print(len(trace))
    

