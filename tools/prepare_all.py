"""Compose pinned AnyText and corrected augmentation, each sampled by parent."""
import fcntl,json,sqlite3,subprocess,sys
from pathlib import Path
from prepare_glyph import ROOT,sha,write
from restore_augmentation import restore

def prepare(args,release):
    assert set(release['components'])=={'anytext_unique','augmentation'}
    for k,c in release['components'].items():
        if c['status']!='released':raise ValueError('Not released: '+k)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    recipe=dict(repo_id=release['repo_id'],revision=release['revision'],component='all',
        indexes={k:v['index']['sha256'] for k,v in release['components'].items()},
        max_shards=args.max_shards,shard_offset=args.shard_offset)
    if (out/'recipe.json').exists():
        if json.loads((out/'recipe.json').read_text())!=recipe:raise ValueError('Use a new output folder for a different data recipe')
    else:write(out/'recipe.json',recipe)
    if (out/'ready.json').exists():
        ready=json.loads((out/'ready.json').read_text())
        assert sha(out/'dataset.sqlite')==ready['dataset_sha256']
        for c in ready['components']:assert sha(out/c['path']/'dataset.sqlite')==c['dataset_sha256']
        print(json.dumps(ready,indent=2));return
    command=[sys.executable,str(ROOT/'tools/prepare_glyph.py'),'--components','anytext_unique',
        '--release',str(args.release.resolve()),'--output',str(out/'anytext_unique'),'--cache',str(args.cache.resolve()),
        '--max-shards',str(args.max_shards),'--shard-offset',str(args.shard_offset)]
    if args.local_snapshot:command+=['--local-snapshot',str(args.local_snapshot.resolve())]
    subprocess.run(command,check=True)
    aug=release['components']['augmentation'];prefix=aug['prefix']
    download=args.local_snapshot.resolve() if args.local_snapshot else args.cache.resolve()/'augmentation_snapshot'/release['revision']
    def fetch(name):
        if args.local_snapshot:return download/name
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(release['repo_id'],name,repo_type='dataset',revision=release['revision'],local_dir=download))
    path=fetch(aug['index']['file']);assert sha(path)==aug['index']['sha256']
    index=json.loads(path.read_text());assert index['complete'] and index['parents']==aug['parents'] and index['crops']==aug['physical_crops']
    meta=fetch(aug['preparation']['file']);assert sha(meta)==aug['preparation']['sha256']
    entries=index['shards'][:args.max_shards] if args.max_shards else index['shards']
    for e in entries:
        p=fetch(prefix+'/'+e['file']);assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
    restored=restore(download/prefix,out/'augmentation',aug['index']['sha256'],args.max_shards or None)
    components=[]
    for name in ['anytext_unique','augmentation']:
        ready=json.loads((out/name/'ready.json').read_text())
        components.append(dict(path=name,images=ready['images'],dataset_sha256=ready['dataset_sha256'],format=ready['format']))
    n=sum(c['images'] for c in components)
    if not args.max_shards:assert n==release['components']['anytext_unique']['images']+aug['parents']
    db=sqlite3.connect(out/'dataset.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS components(path TEXT PRIMARY KEY,images INTEGER,dataset_sha256 TEXT,format TEXT)')
    with db:
        for c in components:db.execute('INSERT OR REPLACE INTO components VALUES(?,?,?,?)',tuple(c[k] for k in ['path','images','dataset_sha256','format']))
    db.close()
    ready=dict(complete=True,format='glyph-combined-parents-v1',images=n,parents=n,
        physical_images=components[0]['images']+restored['physical_crops'],components=components,
        dataset_sha256=sha(out/'dataset.sqlite'),source_release=recipe,smoke_only=bool(args.max_shards),
        parent_sampling_weight=1,variant_policy='AnyText annotation rotation; augmentation crop rotation; one parent weight each',image_bytes_rewritten=False,additional_test_exclusion=False)
    write(out/'ready.json',ready);print(json.dumps(ready,indent=2))
