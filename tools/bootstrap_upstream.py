"""Install only hash-pinned FLUX-Text sources; never download model weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def install(archive=None):
    manifest = json.loads((ROOT / 'provenance/vendored_source.json').read_text())
    target = ROOT / 'vendor/FluxText'
    pending = []
    for row in manifest['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe source manifest path')
        path = target / rel
        if path.is_symlink():
            raise ValueError(f'Refusing source symlink: {rel}')
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Local source was modified; refusing overwrite: {rel}')
        else:
            pending.append(row)
    if not pending:
        return len(manifest['files'])
    if archive:
        data = Path(archive).read_bytes()
    else:
        url = f"https://codeload.github.com/AMAP-ML/FluxText/tar.gz/{manifest['revision']}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    verified = []
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as source:
        for row in pending:
            member = source.getmember('FluxText-' + manifest['revision'] + '/' + row['path'])
            if not member.isfile():
                raise ValueError('Expected a regular upstream file')
            content = source.extractfile(member).read()
            if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError('Upstream source hash mismatch: ' + row['path'])
            verified.append((target / row['path'], content))
    for path, content in verified:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.resolve().is_relative_to(target.resolve()):
            raise ValueError('Upstream destination escapes vendor directory')
        path.write_bytes(content)
    return len(manifest['files'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Optional offline GitHub tar.gz at the pinned revision')
    args = parser.parse_args()
    print(f'Upstream source verified: {install(args.archive)} files')
