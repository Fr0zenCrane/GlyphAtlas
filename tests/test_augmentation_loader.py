"""Parent weighting and synchronized crop identity are release requirements."""
import sys,unittest,tempfile,json,sqlite3,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from data_records import select_record,select_crop

class CropTests(unittest.TestCase):
    def parent(self):
        return dict(key='parent1',parent_sampling_weight=1,crop_variants=[
            dict(image='images/a.png',annotation=dict(img_name='images/a.png',annotations=[dict(text='A'*30)])),
            dict(image='images/b.png',annotation=dict(img_name='images/b.png',annotations=[dict(text='B'*31)]))])
    def test_rotation_is_paired_and_not_truncated(self):
        p=self.parent();seen=[]
        for epoch in range(4):
            annotation,image,is_crop=select_record(p,0,epoch,'WRONG_FIRST_IMAGE.png')
            self.assertTrue(is_crop);self.assertEqual(annotation['img_name'],image)
            self.assertGreater(len(annotation['annotations'][0]['text']),20);seen.append(image)
        self.assertNotEqual(seen[0],seen[1]);self.assertEqual(seen[:2],seen[2:])
        self.assertEqual(select_crop(p,0),select_crop(p,0))
    def test_mismatched_image_is_rejected(self):
        p=self.parent()
        for v in p['crop_variants']:v['annotation']['img_name']='wrong.png'
        with self.assertRaises(ValueError):select_crop(p,0)
    def test_wrong_parent_weight_rejected(self):
        p=self.parent();p['parent_sampling_weight']=2
        with self.assertRaises(ValueError):select_crop(p,0)
    def test_ordinary_parent_unchanged(self):
        p=dict(annotations=[])
        ann,img,aug=select_record(p,1,4,'native.png')
        self.assertIs(ann,p);self.assertEqual(img,'native.png');self.assertFalse(aug)
    def test_missing_font_characters_use_a_real_font(self):
        from font_support import font_path,cmap,BASE
        for text in ['MOBILEᵀᴹ','104⎺','ടെക്\u200cസ്റ്റയിൽസ്','ਹਜਿਃ','𝄞','চপ্পল','ꯃꯅꯤꯄꯨꯔ']:
            import unicodedata
            required={ord(c) for c in text if unicodedata.category(c)[0]!='C'}
            self.assertTrue(required<=cmap(font_path(text)))
        self.assertEqual(font_path('正常中文 ABC 123'),str(BASE))

if __name__=='__main__':unittest.main()
