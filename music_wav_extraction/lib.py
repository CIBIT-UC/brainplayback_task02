import time
import serial

# define wait function
def wait_until(timestamp):
    """
    Function to wait until a certain timestamp (in seconds).
    """
    while time.time() < timestamp:
        time.sleep(0.001)
        
def fade_in(sp,devID,duration,steps,t1):
    """
    Using spotipy volume function to emulate a fade-in.
    
    Args:
        sp: output of spotipy.Spotify
        devID: 'id' field of sp.devices()
        duration: duration of the fade-in in seconds
        steps: number of steps to divide the fade-in into
        t1: output of time.time() just before the fade-in
    """
    
    t_step = duration/steps
    vol_step = int(40/steps)
    
    for ii in range(1,steps+1):
        sp.volume(60+ii*vol_step,device_id=devID)
        wait_until(t1+ii*t_step)

def fade_out(sp,devID,duration,steps,t1):
    """
    Using spotipy volume function to emulate a fade-out.
    
    Args:
        sp: output of spotipy.Spotify
        devID: 'id' field of sp.devices()
        duration: duration of the fade-in in seconds
        steps: number of steps to divide the fade-out into
        t1: output of time.time() just before the fade-in
    """
    
    t_step = duration/steps
    vol_step = int(40/steps)
    
    for ii in range(1,steps+1):
        sp.volume(100-ii*vol_step,device_id=devID)
        wait_until(t1+ii*t_step) 
        
def open_trigger_port():
    """
    Setup a COM port for receiving the trigger from the MRI at ICNAS.
    
    Returns:
        serialPort
    """
    
    #initialise ports
    serialPort = serial.Serial("COM5", baudrate=57600, bytesize=8, parity='N', stopbits=1, timeout=0.0001)

    serialPort.reset_input_buffer();
    serialPort.reset_output_buffer();
    
    return serialPort

def close_trigger_port(serialPort):
    """
    Close the serial port.
    
    Args:
        serialPort: output of serial.Serial()
    """
    serialPort.close()
    
def wait_for_trigger(serialPort,timeout):
    """
    Read the serialPort until a trigger is received, then proceed.
    
    Args:
        serialPort: output of serial.Serial()
        timeout: number of seconds to wait for the trigger
    """
    
    serialPort.reset_input_buffer();
    serialPort.reset_output_buffer();
    
    #time out period
    start_time = time.time() 
    
    print('Checking trigger to start experiment.')

    while time.time()-start_time < timeout:
      nCharsToGet = serialPort.inWaiting()
      if (nCharsToGet)>0:
          # number of incoming bytes.
          message = serialPort.read(nCharsToGet)
          #read the current characters
          print(message)
          break
          
    if (nCharsToGet)==0:
      print("*** TRIGGER NOT FOUND ***")
