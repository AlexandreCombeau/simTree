import logging 
import os
import subprocess
def init():
    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    if not os.path.isdir("logging"):
        subprocess.Popen(["mkdir","logging"]).wait()
    logging.basicConfig(filename="./logging/logging.txt",
                        level=logging.DEBUG,
                        format=LOG_FORMAT)


