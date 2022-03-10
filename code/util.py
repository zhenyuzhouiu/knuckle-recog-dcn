# ========================================================= 
# @ Function File: Util logging function
# =========================================================

import numpy as np
import sys

def Logging(msg, suc=True):
    if suc:
        print ("[*] " + msg)
    else:
        print ("[!] " + msg)