"""Fetch pinned official assets. Never fetch our historical experiment weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor/FluxText'
REV='d406544eed05a4a705693ff8e50b02382879fc79'
BASE_REV='358293da0354175698b67ec8299acf928313a78a'
ANNOT_REV='dcebeaee2f9fdb2876706a9b803b9413408f1f4f'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def verify_source():
    """Verify upstream sources installed by bootstrap_upstream.py."""
    manifest=json.loads((ROOT/'provenance/vendored_source.json').read_text())
    for row in manifest['files']:
        path=VENDOR/row['path']
        assert path.is_file(),f'Missing upstream source: {path}; run tools/bootstrap_upstream.py'
        assert sha(path)==row['sha256'],f'Changed upstream source: {path}'
    return len(manifest['files'])

def download_vendor_assets(fonts_only=False):
    """Fetch only non-code OCR weights/fonts; never clone another code repo."""
    verify_source()
    manifest=json.loads((ROOT/'provenance/vendor_assets.json').read_text())
    for row in manifest['files']:
        if fonts_only and not row['path'].endswith('.ttf'):continue
        path=VENDOR/row['path']
        if path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']:
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        partial=path.with_suffix(path.suffix+'.partial')
        with urllib.request.urlopen(row['url'],timeout=120) as source,partial.open('wb') as output:
            for chunk in iter(lambda:source.read(8*1024**2),b''):output.write(chunk)
        assert partial.stat().st_size==row['bytes'] and sha(partial)==row['sha256'],row['path']
        partial.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-assets',action='store_true')
    p.add_argument('--inference-assets',action='store_true',help='Frozen base components and fonts; no training transformer or OCR weights')
    p.add_argument('--verify-source',action='store_true',help='Offline check of all installed upstream source files')
    p.add_argument('--ocr-assets',action='store_true',help='Only download the pinned OCR weights and fonts')
    p.add_argument('--annotations',action='store_true')
    p.add_argument('--anyword',action='store_true')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--evaluation-assets',action='store_true')
    a=p.parse_args()
    if not any(vars(a).values()):p.error('Select the asset groups to download')
    if a.verify_source:
        print(f'Upstream source verified: {verify_source()} files')
    if a.ocr_assets or a.training_assets:
        download_vendor_assets()
    elif a.inference_assets:
        download_vendor_assets(fonts_only=True)
    if a.training_assets or a.inference_assets or a.annotations or a.anyword:
        from huggingface_hub import hf_hub_download,snapshot_download
    token=os.environ.get('HF_TOKEN') or os.environ.get('GLYPH_OMNI_HF_TOKEN')
    models=Path(os.environ.get('GLYPH_MODEL_ROOT',ROOT/'assets/models'))
    if a.training_assets or a.inference_assets:
        for row in json.loads((ROOT/'provenance/vendor_files.json').read_text())['files']:
            if not a.training_assets and row['path'].endswith(('.pth','.pt')):continue
            assert sha(VENDOR/row['path'])==row['sha256'],row['path']
        base=models/'FLUX.1-Fill-dev-diffusers'
        entries=json.loads((ROOT/'provenance/base_files.json').read_text())['files']
        for row in entries:
            if a.inference_assets and not a.training_assets and row['path'].startswith('transformer/'):continue
            path=Path(hf_hub_download('black-forest-labs/FLUX.1-Fill-dev',row['path'],revision=BASE_REV,local_dir=base,token=token))
            assert path.stat().st_size==row['bytes']
            if row['algorithm']=='sha256':assert sha(path)==row['hash']
            else:
                b=path.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['hash']
    if a.training_assets:
        for name in ['model_512/config.yaml','epoch_100.pt']:
            hf_hub_download('GD-ML/FLUX-Text',name,revision=ANNOT_REV,local_dir=models/'FLUX-Text-official',token=token)
        assert sha(models/'FLUX-Text-official/epoch_100.pt')=='a7e329c97cae19e4fd3ad1b5867036952477dd09e53b90f39b2c68b100060156'
    if a.annotations:
        destination=ROOT/'data/raw/FLUX-Text'
        release=json.loads((ROOT/'provenance/annotation_release.json').read_text())
        for row in release['files']:
            path=hf_hub_download('GD-ML/FLUX-Text',row['path'],revision=ANNOT_REV,local_dir=destination,token=token)
            assert sha(path)==row['local_sha256']
        (destination/'release.json').write_bytes((ROOT/'provenance/annotation_release.json').read_bytes())
    if a.anyword:
        snapshot_download('stzhao/AnyWord-3M',repo_type='dataset',revision='54f67ddf34eb4a5aac46539c2a5b511dd0ca1fc4',
            allow_patterns=['*.parquet','*/*.parquet','README.md'],local_dir=ROOT/'data/raw/AnyWord-3M',token=token)
    if a.benchmark:
        subprocess.run([sys.executable,str(ROOT/'tools/download_full_datasets.py'),'--skip-anyword'],check=True)
        import zipfile
        source=ROOT/'data/raw/AnyText-benchmark/assets/benchmark.zip'
        with zipfile.ZipFile(source) as z:
            target=ROOT/'data/raw/AnyText-benchmark/benchmark_extracted'
            for item in z.infolist():
                path=(target/item.filename).resolve()
                assert path.is_relative_to(target.resolve()),'Unsafe archive member'
            z.extractall(target)
    if a.evaluation_assets:
        from modelscope import snapshot_download as ms_download
        import shutil
        source=Path(ms_download('damo/cv_convnextTiny_ocr-recognition-general_damo'))
        target=ROOT/'data/cache/modelscope/models/damo/cv_convnextTiny_ocr-recognition-general_damo'
        if source.resolve()!=target.resolve():shutil.copytree(source,target,dirs_exist_ok=True)
        torch_home=ROOT/'data/cache/home/.cache/torch'
        os.environ['TORCH_HOME']=str(torch_home)
        import torch
        for url in ['https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
                    'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth']:
            torch.hub.load_state_dict_from_url(url,model_dir=str(torch_home/'hub/checkpoints'),map_location='cpu',check_hash=True)
    if any((a.training_assets,a.inference_assets,a.ocr_assets,a.annotations,a.anyword,a.benchmark,a.evaluation_assets)):
        print('Requested official assets downloaded; applicable pinned hashes checked.')

if __name__=='__main__':main()
