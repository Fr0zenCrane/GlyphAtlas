import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('GLYPH_OMNI_ROOT',str(ROOT))
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'runtime')]
import prepare_data as prep
from data_sources import source_identity
from data_records import select_record
from dataset_contract import require_full_release


class SourceDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.parents=[]
        self.entries=[]
        def image(value):
            im=Image.new('RGB',(512,512),(value,80,90));f=io.BytesIO();im.save(f,format='PNG')
            return f.getvalue(),hashlib.sha256(im.tobytes()).hexdigest()
        def annotation(name,text):
            return dict(img_name=name,width=512,height=512,caption='A sign',annotations=[
                dict(text=text,polygon=[[20,20],[220,20],[220,80],[20,80]],valid=True,language='English')])
        for component in ['anytext','augmentation']:
            path=self.root/(component+'.tar');members=[];parents=[]
            if component=='anytext':
                for value,source in [(1,'Art'),(2,'laion')]:
                    payload,key=image(value);name=key+'.png'
                    p=dict(key=key,image=name,image_sha256=hashlib.sha256(payload).hexdigest(),rgb_sha256=key,
                        width=512,height=512,parent_sampling_weight=1,
                        variants=[dict(variant_id='a',annotation=annotation(name,'ONE'))],
                        aliases=[dict(source=source,variant_id='a',uid=source+'/1')])
                    if source=='Art':
                        p['variants'].append(dict(variant_id='b',annotation=annotation(name,'TWO')))
                        p['aliases'].append(dict(source='laion',variant_id='b',uid='laion/2'))
                    parents.append(p);members.extend([(name,payload),(key+'.json',json.dumps(p).encode())])
            else:
                crops=[]
                for value in (3,4):
                    payload,key=image(value);name='images/'+key+'.png'
                    crops.append(dict(image=name,image_sha256=hashlib.sha256(payload).hexdigest(),rgb_sha256=key,
                        annotation=annotation(name,'CROP'+str(value))))
                    members.append((name,payload))
                p=dict(key='aug-test',source='CTW',parent_sampling_weight=1,parent_rgb_sha256='parent-hash',crop_variants=crops,aliases=[])
                parents.append(p);members.append(('parents/aug-test.json',json.dumps(p).encode()))
            with tarfile.open(path,'w') as tar:
                for name,payload in members:
                    info=tarfile.TarInfo(name);info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
            counts={}
            for p in parents:
                source=source_identity(p)[0];counts[source]=counts.get(source,0)+1
            self.entries.append(dict(file=path.name,component=component,sha256=prep.sha(path),bytes=path.stat().st_size,parents=len(parents),source_parents=counts))
            self.parents.extend(parents)
        self.catalog=dict(schema='glyphatlas-sources-v1',repo_id='example/data',data_revision='pinned',
            assignment='fixture',sources=[dict(id=x) for x in ['art','laion','ctw']],shards=self.entries)
        self.catalog_path=self.root/'catalog.json';prep.write(self.catalog_path,self.catalog)

    def tearDown(self):self.tmp.cleanup()

    def test_source_partition_preserves_shared_tags(self):
        source,tags,raw=source_identity(self.parents[0])
        self.assertEqual((source,tags),('art',['art','laion']))
        chosen=prep.plan(self.catalog,['art'])
        self.assertEqual(chosen['parents'],1);self.assertEqual(chosen['shards'],1)
        self.assertEqual(chosen['download_bytes'],self.entries[0]['bytes'])
        self.assertEqual(prep.plan(self.catalog,['all'])['parents'],3)
        for sources in [['all','art'],['missing'],['art','art']]:
            with self.assertRaises(ValueError):prep.plan(self.catalog,sources)

    def test_restore_variants_bytes_paths_and_restart(self):
        selection=prep.plan(self.catalog,['all'])
        args=SimpleNamespace(output=self.root/'out',catalog=self.catalog_path,max_shards=0,
            local_snapshot=self.root,cache=self.root/'cache')
        ready=prep.prepare(args,self.catalog,selection)
        self.assertEqual((ready['images'],ready['physical_images']),(3,4))
        self.assertEqual(ready['parents_by_source'],dict(art=1,ctw=1,laion=1))
        self.assertEqual(prep.prepare(args,self.catalog,selection),ready)
        with sqlite3.connect(args.output/'dataset.sqlite') as db:
            rows=db.execute('SELECT annotation,image FROM records ORDER BY id').fetchall()
        for (blob,image_path),original in zip(rows,self.parents):
            restored=json.loads(blob)
            self.assertEqual(restored['key'],original['key'])
            self.assertEqual(restored['aliases'],original['aliases'])
            for epoch in [0,1,2,11]:
                old,old_path,aug=select_record(original,0,epoch,original.get('image','unused'))
                new,new_path,new_aug=select_record(restored,0,epoch,image_path)
                self.assertEqual(aug,new_aug)
                self.assertEqual({k:v for k,v in old.items() if k!='img_name'}, {k:v for k,v in new.items() if k!='img_name'})
                self.assertTrue((args.output/new_path).is_file())
                with tarfile.open(self.root/('augmentation.tar' if aug else 'anytext.tar')) as tar:
                    self.assertEqual((args.output/new_path).read_bytes(),tar.extractfile(old_path).read())
        changed=prep.plan(self.catalog,['art'])
        with self.assertRaises(ValueError):prep.prepare(args,self.catalog,changed)

    def test_checksum_and_catalog_mismatch_do_not_commit(self):
        output=self.root/'out';output.mkdir();db=prep.connect(output/'dataset.sqlite')
        entry=prep.plan(self.catalog,['art'])['entries'][0]
        broken=dict(entry,sha256='0'*64)
        with self.assertRaises(ValueError):prep.import_shard(self.root/entry['file'],broken,output,db,{'art'})
        broken=dict(entry,source_parents=dict(art=2))
        with self.assertRaises(ValueError):prep.import_shard(self.root/entry['file'],broken,output,db,{'art'})
        self.assertEqual(db.execute('SELECT count(*) FROM records').fetchone()[0],0)
        self.assertEqual(db.execute('SELECT count(*) FROM imported_shards').fetchone()[0],0)
        db.close()

    def test_full_dataset_guard_rejects_subsets_and_old_revision(self):
        release=dict(revision='new',components=dict(anytext_unique=dict(images=10),augmentation=dict(parents=2)))
        ready=dict(complete=True,smoke_only=False,format='glyph-source-indexed-v1',images=12,
            source_release=dict(revision='new'),parents_by_component=dict(anytext=10,augmentation=2))
        require_full_release(ready,release)
        for patch in [dict(images=10),dict(smoke_only=True),dict(source_release=dict(revision='old')),
                      dict(parents_by_component=dict(anytext=12,augmentation=0))]:
            with self.assertRaises(ValueError):require_full_release(dict(ready,**patch),release)

    def test_source_only_shards_restore_with_native_parent_schema(self):
        catalog=copy.deepcopy(self.catalog)
        catalog['schema']='glyphatlas-source-shards-v1'
        entry=dict(self.entries[1],source='ctw',record_type='crop_views')
        entry.pop('component')
        catalog['shards']=[entry];catalog['sources']=[dict(id='ctw')]
        selection=prep.plan(catalog,['ctw'])
        self.assertEqual(selection['entries'][0]['component'],'augmentation')
        path=self.root/'source-only.json';prep.write(path,catalog)
        args=SimpleNamespace(output=self.root/'source-only',catalog=path,max_shards=0,
            local_snapshot=self.root,cache=self.root/'cache')
        result=prep.prepare(args,catalog,selection)
        self.assertEqual((result['images'],result['physical_images']),(1,2))
        bad=copy.deepcopy(catalog);bad['shards'][0]['source_parents']={'ctw':0,'art':1}
        with self.assertRaises(ValueError):prep.plan(bad,['all'])

    def test_publication_binds_catalog_without_changing_its_bytes(self):
        neutral=copy.deepcopy(self.catalog);neutral.update(repo_id=None,data_revision=None)
        prep.write(self.catalog_path,neutral);original=self.catalog_path.read_bytes()
        release=dict(schema='glyphatlas-training-release-v2',repo_id='owner/published',revision='immutable',
            catalog=dict(sha256=hashlib.sha256(original).hexdigest()))
        path=self.root/'release.json';prep.write(path,release)
        bound=prep.load_catalog(self.catalog_path,path)
        self.assertEqual((bound['repo_id'],bound['data_revision']),('owner/published','immutable'))
        self.assertEqual(self.catalog_path.read_bytes(),original)
        overridden=prep.load_catalog(self.catalog_path,path,'owner/mirror','another-commit')
        self.assertEqual((overridden['repo_id'],overridden['data_revision']),('owner/mirror','another-commit'))
        self.catalog_path.write_bytes(original+b' ')
        with self.assertRaises(ValueError):prep.load_catalog(self.catalog_path,path)

    def test_publication_full_guard_checks_repo_catalog_and_all_sources(self):
        release=dict(schema='glyphatlas-training-release-v2',repo_id='owner/data',revision='commit',catalog=dict(sha256='catalog-sha'),
            record_types=dict(full_image=dict(parents=10),crop_views=dict(parents=2)),
            sources=dict(art=dict(parents=10),ctw=dict(parents=2)))
        identity=dict(repo_id='owner/data',revision='commit',catalog_sha256='catalog-sha',sources=['art','ctw'])
        ready=dict(complete=True,smoke_only=False,format='glyph-source-indexed-v1',images=12,
            source_release=identity,parents_by_component=dict(anytext=10,augmentation=2),parents_by_source=dict(art=10,ctw=2))
        require_full_release(ready,release)
        for patch in [dict(repo_id='wrong/data'),dict(catalog_sha256='wrong'),dict(sources=['art']),dict(revision='wrong')]:
            with self.assertRaises(ValueError):require_full_release(dict(ready,source_release=dict(identity,**patch)),release)
        with self.assertRaises(ValueError):require_full_release(dict(ready,parents_by_source=dict(art=12)),release)


if __name__=='__main__':unittest.main()
