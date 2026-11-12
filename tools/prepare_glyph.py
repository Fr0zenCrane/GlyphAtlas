"""Download a pinned Glyph release and build the actual training database.

No raw-source recleaning, test exclusion, near deduplication or image rewriting.
Interrupted shard imports are transactional; image bytes and all variants remain.
"""
import argparse,fcntl,hashlib,io,json,os,sqlite3,tarfile
from pathlib import Path,PurePosixPath
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def safe_name(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or len(p.parts)!=1 or name in ('.','..') or '\\' in name:raise ValueError(f'Unsafe archive name: {name!r}')
    return name
def validate_parent(parent):
    if parent.get('parent_sampling_weight')!=1:raise ValueError('Expected one sampling weight per parent')
    if parent['width']!=512 or parent['height']!=512:raise ValueError('Release requires final512 images')
    if not parent['variants']:raise ValueError('No variants')
    for variant in parent['variants']:
        a=variant['annotation']
        if a.get('use_re'):raise ValueError('Paired replacement data needs a separate loader')
        if a.get('width')!=512 or a.get('height')!=512 or not a.get('annotations'):raise ValueError('Malformed native annotation')
        if not any(x.get('valid') is not False and x.get('text') and x.get('polygon') for x in a['annotations']):raise ValueError('No valid target text')
    return safe_name(parent['image'])

def import_shard(db,path,entry,out):
    """Validate an entire shard before publishing its records in one transaction."""
    if sha(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:raise ValueError('Shard checksum/size mismatch: '+entry['file'])
    prior=db.execute('SELECT sha256 FROM imported_shards WHERE file=?',(entry['file'],)).fetchone()
    if prior:
        if prior[0]!=entry['sha256']:raise ValueError('Changed shard in resumed import')
        return
    parents=[];images={};seen=set()
    with tarfile.open(path,'r|*') as archive:
        for member in archive:
            name=safe_name(member.name)
            if not member.isfile() or name in seen:raise ValueError('Unexpected/duplicate archive member: '+name)
            seen.add(name);payload=archive.extractfile(member).read()
            if name.endswith('.json'):
                if len(payload)>64*1024**2:raise ValueError('Oversized metadata')
                parents.append(json.loads(payload));continue
            digest=hashlib.sha256(payload).hexdigest()
            with Image.open(io.BytesIO(payload)) as im:
                if im.size!=(512,512):raise ValueError('Unexpected image dimensions')
                rgb=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            relative=f'images/{name[:2]}/{name}';dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if sha(dest)!=digest:raise ValueError('Existing image differs: '+relative)
            else:
                tmp=dest.with_suffix(dest.suffix+'.tmp');tmp.write_bytes(payload);tmp.replace(dest)
            images[name]=(relative,digest,rgb)
    if len(parents)!=entry['images'] or len(images)!=len(parents):raise ValueError('Shard parent/image count mismatch')
    if sum(len(x['variants']) for x in parents)!=entry['annotation_variants']:raise ValueError('Variant count mismatch')
    if sum(len(x['aliases']) for x in parents)!=entry['original_records']:raise ValueError('Alias count mismatch')
    next_id=db.execute('SELECT count(*) FROM records').fetchone()[0];used=set()
    with db:
        for parent in parents:
            name=validate_parent(parent)
            if name in used:raise ValueError('Duplicate image inside parent records')
            used.add(name);relative,digest,rgb=images[name]
            if digest!=parent['image_sha256'] or rgb!=parent['rgb_sha256'] or rgb!=parent['key']:raise ValueError('Per-image checksum mismatch')
            db.execute('INSERT INTO records VALUES(?,?,?,?,?,?)',(next_id,parent['key'],json.dumps(parent,ensure_ascii=False),relative,digest,rgb));next_id+=1
        db.execute('INSERT INTO imported_shards VALUES(?,?,?,?,?)',(entry['file'],entry['sha256'],entry['images'],entry['annotation_variants'],entry['original_records']))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,default=ROOT/'configs/glyph_release.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--components',choices=['anytext_unique','all'],default='anytext_unique')
    p.add_argument('--max-shards',type=int,default=0,help='Smoke only: 0 imports every shard')
    p.add_argument('--shard-offset',type=int,default=0,help='Smoke only; requires --max-shards')
    p.add_argument('--local-snapshot',type=Path,help='Offline snapshot with the pinned metadata and shards')
    args=p.parse_args();release=json.loads(args.release.read_text())
    if release.get('schema')=='glyphatlas-training-release-v2':
        p.error('This release uses source-exclusive shards. Use tools/prepare_data.py --sources all --output ... --cache ... instead.')
    if args.components=='all':
        pending=[k for k,v in release['components'].items() if v['status']!='released']
        if pending:p.error('Full-data delivery is incomplete; components not released: '+', '.join(pending)+'. Nothing was downloaded or silently omitted.')
        unsupported=set(release['components'])-{'anytext_unique','augmentation'}
        if unsupported:p.error('No verified loader for these components: '+', '.join(sorted(unsupported))+'. Publishing files alone does not enable their training.')
    if args.max_shards<0 or args.shard_offset<0 or args.shard_offset and not args.max_shards:p.error('Invalid smoke shard selection')
    if args.components=='all':
        from prepare_all import prepare
        prepare(args,release);return
    component=release['components']['anytext_unique'];out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    recipe=dict(repo_id=release['repo_id'],revision=release['revision'],component='anytext_unique',index_sha256=component['index']['sha256'],max_shards=args.max_shards,shard_offset=args.shard_offset,variant_policy=component['variant_policy'])
    if (out/'recipe.json').exists():
        if json.loads((out/'recipe.json').read_text())!=recipe:raise ValueError('Use a different output directory for a different recipe')
    else:write(out/'recipe.json',recipe)
    if (out/'ready.json').exists():
        ready=json.loads((out/'ready.json').read_text())
        if sha(out/'dataset.sqlite')!=ready['dataset_sha256']:raise ValueError('Completed dataset database changed')
        print(json.dumps(ready,indent=2));return
    def fetch(name):
        if args.local_snapshot:return args.local_snapshot/name
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(release['repo_id'],name,repo_type='dataset',revision=release['revision'],cache_dir=args.cache))
    index_path=fetch(component['index']['file'])
    if sha(index_path)!=component['index']['sha256']:raise ValueError('Pinned index checksum mismatch')
    index=json.loads(index_path.read_text());assert index['complete'] and index['images']==component['images']
    entries=index['shards'][args.shard_offset:]
    if args.max_shards:entries=entries[:args.max_shards]
    if not entries:raise ValueError('Empty shard selection')
    db=sqlite3.connect(out/'dataset.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY,uid TEXT UNIQUE,annotation TEXT,image TEXT,image_sha256 TEXT,rgb_sha256 TEXT UNIQUE);CREATE TABLE IF NOT EXISTS imported_shards(file TEXT PRIMARY KEY,sha256 TEXT,images INT,variants INT,aliases INT);')
    for i,entry in enumerate(entries):
        prior=db.execute('SELECT sha256 FROM imported_shards WHERE file=?',(entry['file'],)).fetchone()
        if prior:
            if prior[0]!=entry['sha256']:raise ValueError('Imported shard version changed')
        else:import_shard(db,fetch(entry['file']),entry,out)
        print(f'{i+1}/{len(entries)} shards verified/imported',flush=True)
    images,variants,aliases=db.execute('SELECT sum(images),sum(variants),sum(aliases) FROM imported_shards').fetchone()
    assert images==db.execute('SELECT count(*) FROM records').fetchone()[0]==sum(e['images'] for e in entries)
    smoke=bool(args.max_shards or args.shard_offset)
    if not smoke:assert (images,variants,aliases)==(component['images'],component['annotation_variants'],component['original_records'])
    db.commit();db.close()
    ready=dict(complete=True,images=images,annotation_variants=variants,original_records=aliases,smoke_only=smoke,dataset_sha256=sha(out/'dataset.sqlite'),format='glyph-parent-variants-v1',source_release=recipe,parent_sampling_weight=1,variant_policy=component['variant_policy'],additional_test_exclusion=False,image_bytes_rewritten=False)
    write(out/'ready.json',ready);print(json.dumps(ready,indent=2))
if __name__=='__main__':main()
