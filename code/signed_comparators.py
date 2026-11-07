"""Counted dense inverse (5) and sparse LDL update (6) on a compressed kernel."""
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from vertex_outage import FLAGS


@lru_cache(maxsize=1)
def library():
    directory=tempfile.TemporaryDirectory(prefix="outage_signed_")
    binary=Path(directory.name)/"signed.so"
    subprocess.run(["g++",*FLAGS,str(Path(__file__).with_suffix(".cpp")),"-o",str(binary)],check=True)
    f=ctypes.CDLL(str(binary)).comparison
    ind=np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    val=np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    cnt=np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS")
    f.argtypes=[ctypes.c_int]*4+[ind,ind,val,val,cnt]
    f.restype=ctypes.c_int
    return f,directory


def compared(n,arrays,mode,audit=True):
    k=len(arrays[2]); values=np.zeros(k); counter=np.zeros(1,np.uint64)
    f,_=library()
    if f(mode,int(audit),n,k,*arrays,values,counter):
        raise ArithmeticError("Signed comparison failed")
    return values,dict(ops=int(counter[0]))
