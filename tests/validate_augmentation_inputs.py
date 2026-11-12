"""Read actual release crops through the native training loader, CPU only."""
import os,sys,json,sqlite3,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from common import sha
from dataset import Dataset,Rows
from data_records import select_crop
import numpy as np,torch
from PIL import Image
from torch.utils.data import DataLoader
torch.set_num_threads(2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ready=json.loads((a.dataset/'ready.json').read_text());assert ready['format']=='glyph-crop-parent-v2'
    db=sqlite3.connect(a.dataset/'dataset.sqlite');d=Dataset(a.dataset);count=0;long=0;paths=set()
    for i,payload in db.execute('SELECT id,annotation FROM records ORDER BY id'):
        parent=json.loads(payload)
        for epoch in range(len(parent['crop_variants'])):
            v=select_crop(parent,epoch);d.data_list.epoch=epoch;info=d.data_list[i]
            assert Path(info['img_path'])==a.dataset/v['image']
            assert sha(Path(info['img_path']))==v['image_sha256'];paths.add(v['image'])
            sample=d[(i,epoch)];n=int(sample['n_lines']);assert 0<n<=8
            texts=sample['texts'][:n];expected={x['text'] for x in v['annotation']['annotations'] if x['valid']}
            assert all(text in expected for text in texts),'GT changed/truncated'
            assert tuple(sample['image'].shape)==(3,512,512)
            assert all(np.isfinite(sample[k]).all() for k in ['image','hint','condition'])
            with Image.open(a.dataset/v['image']) as im:
                expected_image=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1).float()/255
            assert torch.equal(sample['image'],expected_image),'Training image differs from approved crop pixels'
            count+=1;long+=sum(len(t)>20 for t in texts)
    ids=[(i,3) for i in range(min(32,len(d)))]
    aa=DataLoader(d,batch_size=1,sampler=ids,num_workers=0)
    bb=DataLoader(d,batch_size=1,sampler=ids,num_workers=2)
    for x,y in zip(aa,bb):
        for key in ['image','condition','hint','positions','texts']:
            if isinstance(x[key],torch.Tensor):assert torch.equal(x[key],y[key])
            elif isinstance(x[key],list):
                for xx,yy in zip(x[key],y[key]):
                    if isinstance(xx,torch.Tensor):assert torch.equal(xx,yy)
                    else:assert xx==yy
            else:assert x[key]==y[key]
    result=dict(passed=True,parents=len(d),native_inputs=count,unique_physical_crops=len(paths),selected_long_texts_preserved=long,worker_count_parity=[0,2],training_updates=0)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
