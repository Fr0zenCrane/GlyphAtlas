"""Read-only checks of a prepared full-corpus SQLite dataset."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
from PIL import Image

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--all-images',action='store_true');a=p.parse_args();root=a.dataset.resolve()
    ready=json.loads((root/'ready.json').read_text());assert ready['complete']
    assert sha(root/'dataset.sqlite')==ready['dataset_sha256']
    if ready.get('format')=='glyph-combined-parents-v1':
        import subprocess,sys
        assert ready['images']==sum(c['images'] for c in ready['components'])
        for c in ready['components']:
            child=root/c['path'];assert sha(child/'dataset.sqlite')==c['dataset_sha256']
            command=[sys.executable,__file__,'--dataset',str(child)]
            if a.all_images:command+=['--all-images']
            subprocess.run(command,check=True)
        print(json.dumps(dict(passed=True,parents=ready['images'],physical_images=ready['physical_images'],components=ready['components'],smoke_only=ready['smoke_only']),indent=2));return
    db=sqlite3.connect(f'file:{root}/dataset.sqlite?mode=ro',uri=True)
    n,lo,hi=db.execute('SELECT count(*),min(id),max(id) FROM records').fetchone()
    assert n==ready['images'] and lo==0 and hi==n-1
    ids=range(n) if a.all_images else sorted({i*(n-1)//31 for i in range(32)})
    checked=0
    for i in ids:
        path,digest,annotation=db.execute('SELECT image,image_sha256,annotation FROM records WHERE id=?',(i,)).fetchone()
        path=(root/path).resolve();assert path.is_relative_to(root)
        assert sha(path)==digest
        with Image.open(path) as im:assert im.size==(512,512)
        row=json.loads(annotation)
        variants=row.get('crop_variants',row.get('variants',[{'annotation':row}]))
        assert variants
        for v in variants:
            assert v['annotation']['annotations'] and not v['annotation'].get('use_re')
            if 'crop_variants' in row:
                p=(root/v['image']).resolve();assert p.is_relative_to(root)
                assert v['annotation']['img_name']==v['image'] and sha(p)==v['image_sha256']
                with Image.open(p) as im:assert im.size==(512,512)
        checked+=1
    print(json.dumps(dict(passed=True,images=n,images_hash_checked=checked,global_batch=128,
        updates_per_epoch=n//128,drop_last_per_epoch=n%128,smoke_only=ready['smoke_only']),indent=2))

if __name__=='__main__':main()
