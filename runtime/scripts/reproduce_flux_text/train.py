"""Path/I/O compatibility shim for the frozen checkpoint functions."""
import os,sys
from pathlib import Path
from common import ROOT,VENDOR as CODE,sha,write
sys.path.insert(0,str(CODE))
