"""Eight-rank native FLUX-Text generation for all 2K official benchmark inputs.

Uses the public inference class/renderer and one output per source as released.
The paper says batch size four but that is not implemented in the release; this
choice is explicit and the resulting metrics must be labeled accordingly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT=Path(os.environ["GLYPH_OMNI_ROOT"])
MODELS=Path(os.environ.get('GLYPH_MODEL_ROOT',str(ROOT/'assets/models')))
CODE=ROOT/'vendor/FluxText'
BENCH=Path(os.environ.get('GLYPH_BENCHMARK_ROOT',str(ROOT/'data/raw/AnyText-benchmark/benchmark_extracted/benchmark')))
OUT=Path(os.environ.get('FLUX_TEXT_RUN_DIR',str(ROOT/'outputs/flux-text-reproduction'))).resolve()
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'eval'))
import numpy as np
import torch
import torch.distributed as dist
import yaml
from safetensors.torch import load_file
from eval.anytext_inference import T3DataSetWarp
from src.train.model import OminiModelFIll
from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill


class BenchmarkDataset(T3DataSetWarp):
    def __init__(self,*args,**kwargs):
        assert kwargs.get('use_filter',False) is False
        super().__init__(*args,**kwargs)

    def isvalid_img(self,anns):
        # The public benchmark has no training-only recog_valid fields. This
        # metadata is read unconditionally upstream but never used when
        # use_filter=False. Bypass that unused check, retaining annotation.valid,
        # native rendering and every original benchmark image without filtering.
        return True


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
