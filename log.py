import logging 
import os
import subprocess
from datetime import datetime

def init():
    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    if not os.path.isdir("logging"):
        subprocess.Popen(["mkdir","logging"]).wait()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M")
    logging.basicConfig(filename="./logging/log_"+dt_string+".log",
                        level=logging.INFO,
                        format=LOG_FORMAT)


