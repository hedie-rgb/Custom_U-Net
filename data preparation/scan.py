

import os
import glob
import numpy as np

def available_fname(format, reserved_space = 4):
    "resrved_space: how much space to reserve for mumbers (for filling with zeros form left)"

    ind1 = format.find('*')
    ind2 = ind1-len(format)+1
    
    l = [int(s[ind1:ind2]) for s in glob.glob(format)]
        
    i = np.max(l)+1 if len(l) > 0 else 1
    fill = str(i).rjust(reserved_space, '0')

    fname = format[:ind1] + fill + format[ind2:]

    return fname, fill



for i in range(2):

    fname, num = available_fname('scan_*.png')

    #os.system("scanimage -x 210 -y 148 --format png > %s"%fname)

    os.system("scanimage  --format png > %s"%fname)


    if (os.path.getsize(fname) < 1000):
        os.remove(fname)
    else:
        os.system("gio open %s"%fname)
        break





