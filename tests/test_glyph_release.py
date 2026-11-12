"""Release integrity, restart, parent weighting and annotation-variant tests."""
import hashlib,importlib.util,io,json,sqlite3,sys,tarfile,tempfile,unittest
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from data_records import select_annotation
spec=importlib.util.spec_from_file_location('prepare_glyph',ROOT/'tools/prepare_glyph.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)

class ReleaseTests(unittest.TestCase):
    def make_shard(self,root,corrupt=False):
        im=Image.new('RGB',(512,512),(12,34,56));f=io.BytesIO();im.save(f,format='PNG');payload=f.getvalue();key=hashlib.sha256(im.tobytes()).hexdigest()
        def annotation(text):return dict(width=512,height=512,caption='a sign',annotations=[dict(text=text,polygon=[[0,0],[100,0],[100,50],[0,50]],valid=True,language='Latin')])
        parent=dict(key=key,image=key+'.png',image_sha256=hashlib.sha256(payload).hexdigest(),rgb_sha256=key,width=512,height=512,parent_sampling_weight=1,variants=[dict(variant_id='a',annotation=annotation('ONE')),dict(variant_id='b',annotation=annotation('TWO'))],aliases=[dict(uid=str(i),variant_id='a' if i<5 else 'b') for i in range(8)])
        if corrupt:parent['image_sha256']='0'*64
        path=root/'test.tar'
        with tarfile.open(path,'w') as t:
            for name,data in [(parent['image'],payload),(key+'.json',json.dumps(parent).encode())]:
                info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
        return path,dict(file='test.tar',sha256=prep.sha(path),bytes=path.stat().st_size,images=1,annotation_variants=2,original_records=8),parent
    def db(self,root):
        db=sqlite3.connect(root/'dataset.sqlite');db.executescript('CREATE TABLE records(id INTEGER PRIMARY KEY,uid TEXT UNIQUE,annotation TEXT,image TEXT,image_sha256 TEXT,rgb_sha256 TEXT UNIQUE);CREATE TABLE imported_shards(file TEXT PRIMARY KEY,sha256 TEXT,images INT,variants INT,aliases INT);');return db
    def test_parent_once_variants_preserved_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);path,entry,parent=self.make_shard(r);db=self.db(r)
            prep.import_shard(db,path,entry,r);prep.import_shard(db,path,entry,r)
            self.assertEqual(db.execute('SELECT count(*) FROM records').fetchone()[0],1)
            restored=json.loads(db.execute('SELECT annotation FROM records').fetchone()[0]);self.assertEqual(restored,parent)
            self.assertNotEqual(select_annotation(restored,0,0),select_annotation(restored,0,1))
            self.assertEqual(select_annotation(restored,0,0),select_annotation(restored,0,2))
            self.assertEqual(select_annotation(restored,0,1),select_annotation(restored,0,1))
    def test_corrupt_image_rolls_back(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);path,entry,parent=self.make_shard(r,True);db=self.db(r)
            with self.assertRaises(ValueError):prep.import_shard(db,path,entry,r)
            self.assertEqual(db.execute('SELECT count(*) FROM records').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT count(*) FROM imported_shards').fetchone()[0],0)
    def test_shard_hash_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);path,entry,parent=self.make_shard(r);db=self.db(r);entry['sha256']='0'*64
            with self.assertRaises(ValueError):prep.import_shard(db,path,entry,r)
    def test_unsafe_names(self):
        for name in ['../escape','/tmp/escape','a/b','a\\b','..','']:
            with self.assertRaises(ValueError):prep.safe_name(name)
    def test_old_records_unchanged(self):
        record={'annotations':[]};self.assertIs(select_annotation(record,3,100),record)
if __name__=='__main__':unittest.main()
