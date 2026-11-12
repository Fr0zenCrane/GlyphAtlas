"""Build a source-only archive with per-file hashes; exclude installed assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {'runtime','tools','configs','tests','docs','examples','provenance','assets'}
ROOT_FILES = {'README.md','NOTICE','LICENSE','.gitignore','.gitattributes',
              'requirements.txt','requirements.lock.txt'}


def source_files():
    files=[]
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if not path.is_file() or '__pycache__' in rel.parts:continue
        if rel.parts[0] not in DIRECTORIES and str(rel) not in ROOT_FILES:continue
        if rel.parts[0]=='assets' and rel.parts[1]!='fonts':continue
        if path.suffix in {'.pyc','.pyo','.pt','.pth','.ckpt','.safetensors','.sqlite','.zip'}:continue
        if str(rel)=='provenance/release_files.json':continue
        if path.is_symlink():raise ValueError('Source archive cannot contain symlinks: '+str(rel))
        if path.suffix not in {'.ttf','.png'}:
            content=path.read_text()
            forbidden=[r'/vePFS[-/]',r'/root/\.ssh',r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----',
                       r'\b(?:hf_|ghp_|github_pat_|olp_)[A-Za-z0-9_]{16,}',r'192\.168\.\d+\.\d+']
            if any(re.search(pattern,content) for pattern in forbidden):
                raise ValueError('Possible private path or credential in '+str(rel))
        files.append(path)
    return files


def build(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new archive path')
    files=source_files()
    manifest=dict(format='glyphatlas-source-v1',files=[dict(path=str(p.relative_to(ROOT)),
        bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    payload=(json.dumps(manifest,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,'GlyphAtlas/'+str(path.relative_to(ROOT)))
        archive.writestr('GlyphAtlas/provenance/release_files.json',payload)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise RuntimeError('Archive CRC failure')
        for row in manifest['files']:
            if hashlib.sha256(archive.read('GlyphAtlas/'+row['path'])).hexdigest()!=row['sha256']:
                raise RuntimeError('Archive member mismatch: '+row['path'])
    print(json.dumps(dict(files=len(files)+1,bytes=output.stat().st_size,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
