"""Download the published corpus by source and restore a training-ready database."""
import argparse
import collections
import copy
import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile

from PIL import Image
from data_sources import source_identity, SOURCE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8<<20),b''):digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_catalog(path,release_path=None,repo_id=None,revision=None):
    """Bind the byte-identical portable catalog to the pinned publication."""
    path=Path(path);catalog=read(path)
    if release_path is not None:
        release=read(release_path)
        if release.get('schema')=='glyphatlas-training-release-v2':
            if sha(path)!=release['catalog']['sha256']:
                raise ValueError('Bundled source catalog differs from the pinned publication')
            catalog['repo_id']=release['repo_id']
            catalog['data_revision']=release['revision']
    if repo_id:catalog['repo_id']=repo_id
    if revision:catalog['data_revision']=revision
    return catalog


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def safe_name(name):
    path=PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or str(path)!=name:
        raise ValueError('Unsafe archive path: '+repr(name))
    return name


def plan(catalog,sources,max_shards=0):
    if catalog['schema'] not in ('glyphatlas-sources-v1','glyphatlas-source-shards-v1'):
        raise ValueError('Unsupported source catalog')
    available={s['id'] for s in catalog['sources']}
    if not available<=set(SOURCE_NAMES):raise ValueError('Unknown catalog sources')
    if sources==['all']:selected=sorted(available)
    else:
        if 'all' in sources or not sources or len(sources)!=len(set(sources)):
            raise ValueError('Use all, or distinct source IDs')
        selected=sorted(sources)
    unknown=set(selected)-available
    if unknown:raise ValueError('Unknown source IDs: '+', '.join(sorted(unknown)))
    if max_shards<0:raise ValueError('max-shards must be nonnegative')
    entries=[]
    for entry in catalog['shards']:
        if catalog['schema']=='glyphatlas-source-shards-v1':
            kinds={'full_image':'anytext','crop_views':'augmentation'}
            if entry.get('record_type') not in kinds:raise ValueError('Unknown parent record type')
            if entry.get('source_parents')!={entry.get('source'):entry['parents']}:
                raise ValueError('Expected exactly one source per TAR')
            entry=dict(entry,component=kinds[entry['record_type']])
        safe_name(entry['file'])
        if entry['component'] not in ('anytext','augmentation'):raise ValueError('Unknown component')
        if sum(entry['source_parents'].values())!=entry['parents']:raise ValueError('Invalid source partition')
        n=sum(entry['source_parents'].get(source,0) for source in selected)
        if n:entries.append(dict(entry,selected_parents=n))
    if max_shards:entries=entries[:max_shards]
    if not entries:raise ValueError('No shards selected')
    return dict(repo_id=catalog['repo_id'],revision=catalog['data_revision'],sources=selected,
        parents=sum(e['selected_parents'] for e in entries),shards=len(entries),
        download_bytes=sum(e['bytes'] for e in entries),smoke_only=bool(max_shards),entries=entries)


def connect(path):
    db=sqlite3.connect(path)
    db.executescript('''CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY,uid TEXT UNIQUE,annotation TEXT,image TEXT,image_sha256 TEXT,
        rgb_sha256 TEXT,source TEXT,component TEXT);
        CREATE TABLE IF NOT EXISTS imported_shards(file TEXT PRIMARY KEY,sha256 TEXT,parents INT,physical_images INT);
        CREATE INDEX IF NOT EXISTS source_lookup ON records(source);
    ''')
    return db


def import_shard(path,entry,output,db,selected):
    previous=db.execute('SELECT sha256,parents,physical_images FROM imported_shards WHERE file=?',(entry['file'],)).fetchone()
    if previous:
        if previous[:2]!=(entry['sha256'],entry['selected_parents']):raise ValueError('Imported shard identity changed')
        return previous[1],previous[2]
    path,output=Path(path),Path(output)
    if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
        raise ValueError('Shard checksum/size mismatch: '+entry['file'])
    with tarfile.open(path,'r:') as archive:
        members={};parents=[]
        for member in archive:
            safe_name(member.name)
            if not member.isfile() or member.name in members:raise ValueError('Duplicate or non-file archive member')
            members[member.name]=member
            if member.name.endswith('.json'):
                if member.size>64<<20:raise ValueError('Oversized parent record')
                parents.append(json.loads(archive.extractfile(member).read()))
        if len(parents)!=entry['parents']:raise ValueError('Canonical parent count mismatch')
        source_counts=collections.Counter(source_identity(p)[0] for p in parents)
        if dict(source_counts)!=entry['source_parents']:raise ValueError('Source catalog does not match the actual shard')
        chosen=[p for p in parents if source_identity(p)[0] in selected]
        if len(chosen)!=entry['selected_parents']:raise ValueError('Selected parent count mismatch')
        count=0;physical=0
        with db:
            next_id=db.execute('SELECT count(*) FROM records').fetchone()[0]
            for original in chosen:
                parent=copy.deepcopy(original)
                source,tags,_=source_identity(parent)
                augmentation=entry['component']=='augmentation'
                if augmentation!=('crop_variants' in parent) or parent.get('parent_sampling_weight')!=1:
                    raise ValueError('Parent sampling contract mismatch')
                crops=parent.get('crop_variants',[])
                if augmentation and not 1<=len(crops)<=2:raise ValueError('Invalid crop count')
                if not augmentation and not parent.get('variants'):raise ValueError('Missing annotation variants')
                image_rows=crops if augmentation else [parent]
                copied=[]
                for image_row in image_rows:
                    name=safe_name(image_row['image'])
                    member=members.get(name)
                    if member is None or name.endswith('.json'):raise ValueError('Missing image member')
                    payload=archive.extractfile(member).read()
                    digest=hashlib.sha256(payload).hexdigest()
                    if digest!=image_row['image_sha256']:raise ValueError('Image byte hash mismatch')
                    with Image.open(io.BytesIO(payload)) as image:
                        if image.size!=(512,512):raise ValueError('Expected a final 512px image')
                        rgb=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
                    if rgb!=image_row['rgb_sha256']:raise ValueError('Decoded RGB hash mismatch')
                    record_type='crop_views' if augmentation else 'full_image'
                    relative=f'images/{source}/{record_type}/{Path(name).name}'
                    destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                    if not destination.resolve().is_relative_to(output.resolve()):raise ValueError('Output image path escapes dataset')
                    if destination.exists():
                        if sha(destination)!=digest:raise ValueError('Existing output image differs')
                    else:
                        temporary=destination.with_suffix('.tmp');temporary.write_bytes(payload);temporary.replace(destination)
                    copied.append((relative,digest,rgb))
                    if augmentation:
                        if image_row['annotation']['img_name']!=name:raise ValueError('Crop annotation/image mismatch')
                        image_row['image']=relative;image_row['annotation']['img_name']=relative
                    else:
                        if parent['key']!=rgb:raise ValueError('AnyText parent identity mismatch')
                        parent['image']=relative
                        for variant in parent['variants']:variant['annotation']['img_name']=relative
                # Preserve keys, variants, aliases, polygons and captions. Only
                # portable image paths change when restoring into source folders.
                first=copied[0]
                db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?)',
                    (next_id,parent['key'],json.dumps(parent,ensure_ascii=False),first[0],first[1],
                     parent['parent_rgb_sha256'] if augmentation else first[2],source,entry['component']))
                next_id+=1;count+=1;physical+=len(copied)
            db.execute('INSERT INTO imported_shards VALUES(?,?,?,?)',(entry['file'],entry['sha256'],count,physical))
    return count,physical


def prepare(args,catalog,selection):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return prepare_locked(args,catalog,selection,output)


def prepare_locked(args,catalog,selection,output):
    identity=dict(schema='glyphatlas-source-restore-v2',catalog_sha256=sha(args.catalog),
        repo_id=selection['repo_id'],revision=selection['revision'],sources=selection['sources'],
        max_shards=args.max_shards,source_assignment=catalog['assignment'],image_layout='source-record-type-v1')
    recipe=output/'recipe.json'
    if recipe.exists():
        if read(recipe)!=identity:raise ValueError('Different data selection/revision: use a fresh output directory')
    elif (output/'dataset.sqlite').exists():
        raise ValueError('Unbound existing database: use a fresh output directory')
    else:write(recipe,identity)
    if (output/'ready.json').exists():
        ready=read(output/'ready.json')
        if sha(output/'dataset.sqlite')!=ready['dataset_sha256']:raise ValueError('Prepared database changed')
        return ready
    db=connect(output/'dataset.sqlite')
    try:
        for number,entry in enumerate(selection['entries'],1):
            prior=db.execute('SELECT sha256 FROM imported_shards WHERE file=?',(entry['file'],)).fetchone()
            if prior:
                if prior[0]!=entry['sha256']:raise ValueError('Imported shard identity changed')
                continue
            if args.local_snapshot:
                payload=args.local_snapshot/entry['file']
            else:
                from huggingface_hub import hf_hub_download
                payload=Path(hf_hub_download(selection['repo_id'],entry['file'],repo_type='dataset',
                    revision=selection['revision'],cache_dir=args.cache))
            parents,physical=import_shard(payload,entry,output,db,set(selection['sources']))
            write(output/'progress.json',dict(completed_shards=db.execute('SELECT count(*) FROM imported_shards').fetchone()[0],
                total_shards=selection['shards'],restored_parents=db.execute('SELECT count(*) FROM records').fetchone()[0]))
            print(f'[{number}/{selection["shards"]}] {entry["file"]}: {parents} selected parents, {physical} images',flush=True)
        n=db.execute('SELECT count(*) FROM records').fetchone()[0]
        if n!=selection['parents']:raise ValueError('Final dataset count mismatch')
        physical=db.execute('SELECT sum(physical_images) FROM imported_shards').fetchone()[0]
        counts={source:count for source,count in db.execute('SELECT source,count(*) FROM records GROUP BY source')}
        components={component:count for component,count in db.execute('SELECT component,count(*) FROM records GROUP BY component')}
        db.commit()
    finally:db.close()
    ready=dict(complete=True,format='glyph-source-indexed-v1',images=n,parents=n,physical_images=physical,
        smoke_only=selection['smoke_only'],dataset_sha256=sha(output/'dataset.sqlite'),source_release=identity,
        parents_by_source=counts,parents_by_component=components,
        parents_by_record_type={'full_image':components.get('anytext',0),'crop_views':components.get('augmentation',0)},parent_sampling_weight=1,
        variant_policy='Rotate annotation or crop variant within each parent; no source reweighting',
        image_bytes_rewritten=False,additional_test_exclusion=False)
    write(output/'ready.json',ready)
    return ready


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,default=ROOT/'configs/data_sources.json')
    parser.add_argument('--repo-id',help='Bind a repository-neutral catalog to the new dataset repository')
    parser.add_argument('--revision',help='Pin the commit of the new dataset repository')
    parser.add_argument('--sources',nargs='+',default=['all'],help='all, or source IDs printed by --list-sources')
    parser.add_argument('--list-sources',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'data/glyphatlas')
    parser.add_argument('--cache',type=Path,default=ROOT/'data/hf_cache')
    parser.add_argument('--local-snapshot',type=Path,help='Offline copy of the pinned canonical shard paths')
    parser.add_argument('--max-shards',type=int,default=0,help='First N covering shards for smoke tests only; 0 means all')
    parser.add_argument('--dry-run',action='store_true',help='Show exact parent/download counts without downloading')
    args=parser.parse_args()
    default_catalog=ROOT/'configs/data_sources.json'
    release_path=ROOT/'configs/glyph_release.json' if args.catalog.resolve()==default_catalog.resolve() else None
    catalog=load_catalog(args.catalog,release_path,args.repo_id,args.revision)
    if args.list_sources:
        for source in catalog['sources']:
            print(f'{source["id"]:20} {source["parents"]:>9,} parents  {source["name"]}')
        return
    selection=plan(catalog,args.sources,args.max_shards)
    if not args.dry_run and not args.local_snapshot and not (selection['repo_id'] and selection['revision']):
        parser.error('Bind this catalog with --repo-id and --revision, or supply --local-snapshot for offline restoration')
    print(json.dumps({k:v for k,v in selection.items() if k!='entries'},indent=2))
    if not args.dry_run:print(json.dumps(prepare(args,catalog,selection),indent=2))


if __name__=='__main__':main()
