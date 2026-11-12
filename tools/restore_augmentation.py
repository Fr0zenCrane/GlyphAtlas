"""Restore a downloaded corrected augmentation release without changing pixels.

python restore.py --snapshot augmentation_corrected_v2 --output /shared/augmentation \
  --index-sha256 <published-shard-index-sha256>

The database has one record per original parent. Use select_crop(parent, epoch),
not one training record per physical image. No training is launched here.
"""
import argparse,hashlib,json,sqlite3,tarfile,io,os
from pathlib import Path
from PIL import Image

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def select_crop(parent,epoch):
 assert parent['parent_sampling_weight']==1 and 1<=len(parent['crop_variants'])<=2
 offset=int.from_bytes(hashlib.sha256(parent['key'].encode()).digest()[:8],'big')
 return parent['crop_variants'][(offset+int(epoch))%len(parent['crop_variants'])]

def safe_path(root,name):
 p=(root/name).resolve()
 if not p.is_relative_to(root.resolve()) or Path(name).is_absolute():raise ValueError('Unsafe member path')
 return p

def restore(snapshot,output,index_sha256,max_shards=None):
 snapshot,output=Path(snapshot).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
 if sha(snapshot/'shard_index.json')!=index_sha256:raise ValueError('Index checksum mismatch')
 index=json.loads((snapshot/'shard_index.json').read_text());assert index['complete'] and index['schema']=='glyph_corrected_augmentation_v2'
 db=sqlite3.connect(output/'dataset.sqlite');db.execute('PRAGMA journal_mode=WAL')
 db.executescript('CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY,uid TEXT UNIQUE,annotation TEXT,image TEXT,image_sha256 TEXT,rgb_sha256 TEXT UNIQUE);CREATE TABLE IF NOT EXISTS shards(file TEXT PRIMARY KEY,sha256 TEXT,parents INT,crops INT);CREATE TABLE IF NOT EXISTS release_identity(sha256 TEXT PRIMARY KEY);')
 prior=db.execute('SELECT sha256 FROM release_identity').fetchall()
 if prior and prior!=[(index_sha256,)]:raise ValueError('Refusing mixed dataset releases')
 db.execute('INSERT OR IGNORE INTO release_identity VALUES(?)',(index_sha256,));db.commit()
 if (output/'ready.json').exists():
  ready=json.loads((output/'ready.json').read_text());assert sha(output/'dataset.sqlite')==ready['dataset_sha256'];return ready
 for entry in index['shards'][:max_shards]:
  old=db.execute('SELECT sha256 FROM shards WHERE file=?',(entry['file'],)).fetchone()
  if old:
   assert old[0]==entry['sha256'];continue
  p=safe_path(snapshot,entry['file']);assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
  parents=[];images={}
  with tarfile.open(p) as tar:
   names=set()
   for member in tar:
    assert member.isfile() and member.name not in names;names.add(member.name);safe_path(output,member.name)
    content=tar.extractfile(member).read()
    if member.name.startswith('parents/') and member.name.endswith('.json'):parents.append(json.loads(content))
    elif member.name.startswith('images/') and member.name.endswith('.png'):
     with Image.open(io.BytesIO(content)) as im:
      assert im.mode=='RGB' and im.size==(512,512);rgb=hashlib.sha256(im.tobytes()).hexdigest()
     dest=safe_path(output,member.name);dest.parent.mkdir(parents=True,exist_ok=True)
     tmp=dest.with_suffix('.tmp');tmp.write_bytes(content);tmp.replace(dest)
     images[member.name]=(hashlib.sha256(content).hexdigest(),rgb)
    else:raise ValueError('Unexpected member '+member.name)
  assert len(parents)==entry['parents'] and len(images)==entry['crops'];used=set()
  with db:
   next_id=db.execute('SELECT count(*) FROM records').fetchone()[0]
   for parent in parents:
    assert parent['parent_sampling_weight']==1 and 1<=len(parent['crop_variants'])<=2
    for crop in parent['crop_variants']:
     assert images[crop['image']]==(crop['image_sha256'],crop['rgb_sha256']);assert crop['image'] not in used;used.add(crop['image'])
     assert crop['annotation']['img_name']==crop['image']
     for a in crop['annotation']['annotations']:
      if a['valid']:assert a['text'] and a['output_char_measure']['px']>=20 and a['gt_preparation_verified']
    first=parent['crop_variants'][0]
    db.execute('INSERT INTO records VALUES(?,?,?,?,?,?)',(next_id,parent['key'],json.dumps(parent,ensure_ascii=False),first['image'],first['image_sha256'],parent['parent_rgb_sha256']));next_id+=1
   assert used==set(images)
   db.execute('INSERT INTO shards VALUES(?,?,?,?)',(entry['file'],entry['sha256'],entry['parents'],entry['crops']))
 n,crops=db.execute('SELECT sum(parents),sum(crops) FROM shards').fetchone();assert n==db.execute('SELECT count(*) FROM records').fetchone()[0]
 if max_shards is None:assert n==index['parents'] and crops==index['crops']
 db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
 ready=dict(complete=True,images=n,parents=n,physical_crops=crops,smoke_only=max_shards is not None,format='glyph-crop-parent-v2',parent_sampling_weight=1,variant_policy='rotate_one_crop_per_parent_per_epoch',dataset_sha256=sha(output/'dataset.sqlite'),source_index_sha256=index_sha256,image_bytes_rewritten=False)
 (output/'ready.json').write_text(json.dumps(ready,indent=2));return ready

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True);p.add_argument('--index-sha256',required=True);p.add_argument('--max-shards',type=int);a=p.parse_args()
 print(json.dumps(restore(a.snapshot,a.output,a.index_sha256,a.max_shards),indent=2))
