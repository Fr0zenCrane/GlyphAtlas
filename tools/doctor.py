"""Read-only receiving-machine checks. Never launches training or downloads."""
import argparse,importlib.metadata,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'tools')]
from download_assets import verify_source,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path);p.add_argument('--training-assets',action='store_true');p.add_argument('--gpu',action='store_true');p.add_argument('--require-all-data',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.require_all_data and not a.dataset:p.error('--require-all-data requires --dataset; release metadata alone is not a loading check')
    result={'passed':True,'checks':{},'errors':[],'warnings':[]}
    def check(name,fn):
        try:result['checks'][name]=fn()
        except Exception as e:result['passed']=False;result['errors'].append(name+': '+str(e))
    check('bundled_source',verify_source)
    def versions():
        expected={'torch':'2.4.0','torchvision':'0.19.0','diffusers':'0.32.2','transformers':'4.46.3'}
        actual={k:importlib.metadata.version(k) for k in expected}
        assert all(actual[k].split('+')[0]==v for k,v in expected.items()),actual
        import torch
        assert torch.version.cuda=='12.1',torch.version.cuda
        assert sys.version_info[:2]==(3,10),sys.version
        return actual
    check('runtime_versions',versions)
    def fonts():
        entries=json.loads((ROOT/'provenance/fallback_fonts.json').read_text())['files']
        for row in entries:assert sha(ROOT/row['file'])==row['sha256'],row['file']
        return len(entries)
    check('fallback_font_assets',fonts)
    release=json.loads((ROOT/'configs/glyph_release.json').read_text())
    released_groups=release['sources'] if release.get('schema')=='glyphatlas-training-release-v2' else release['components']
    pending=[k for k,v in released_groups.items() if v['status']!='released']
    result['checks']['data_release']={'repo':release['repo_id'],'revision':release['revision'],'unreleased_components':pending}
    if release.get('schema')=='glyphatlas-training-release-v2':
        def catalog():
            path=ROOT/'configs/data_sources.json'
            assert sha(path)==release['catalog']['sha256'], 'Bundled catalog hash differs from the pinned release'
            content=json.loads(path.read_text())
            assert content['parents']==release['parents'] and len(content['shards'])==release['shards']
            return {'sha256':release['catalog']['sha256'],'sources':len(content['sources']),'parents':content['parents'],'shards':len(content['shards'])}
        check('pinned_source_catalog',catalog)
    if pending:
        (result['errors'] if a.require_all_data else result['warnings']).append('Not released: '+', '.join(pending))
        if a.require_all_data:result['passed']=False
    if a.training_assets:
        def assets():
            for row in json.loads((ROOT/'provenance/vendor_assets.json').read_text())['files']:
                assert sha(ROOT/'vendor/FluxText'/row['path'])==row['sha256'],row['path']
            models=Path(os.environ.get('GLYPH_MODEL_ROOT',ROOT/'assets/models'))
            for row in json.loads((ROOT/'provenance/base_files.json').read_text())['files']:
                f=models/'FLUX.1-Fill-dev-diffusers'/row['path'];assert f.stat().st_size==row['bytes'],str(f)
                if row['algorithm']=='sha256':assert sha(f)==row['hash'],str(f)
                else:
                    import hashlib
                    b=f.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['hash']
            assert sha(models/'FLUX-Text-official/epoch_100.pt')=='a7e329c97cae19e4fd3ad1b5867036952477dd09e53b90f39b2c68b100060156'
            assert (models/'FLUX-Text-official/model_512/config.yaml').is_file()
            return 'all pinned training asset hashes verified'
        check('training_assets',assets)
    if a.dataset:
        def dataset():
            from dataset import Dataset
            ready=json.loads((a.dataset/'ready.json').read_text());assert ready['complete']
            if a.require_all_data:
                from dataset_contract import require_full_release
                require_full_release(ready,release)
            assert sha(a.dataset/'dataset.sqlite')==ready['dataset_sha256']
            if ready.get('format')=='glyph-source-indexed-v1':
                recipe=json.loads((a.dataset/'recipe.json').read_text())
                assert ready['source_release']==recipe, 'Prepared source selection differs from its saved recipe'
                assert recipe.get('catalog_sha256'), 'Missing source catalog identity'
            elif ready.get('source_release'):
                assert ready['source_release']['revision']==release['revision'], 'Dataset revision differs from the pinned legacy release'
            d=Dataset(a.dataset);rows=[]
            for i in sorted({0,len(d)//2,len(d)-1}):
                x=d[(i,0)];assert tuple(x['image'].shape)==(3,512,512) and int(x['n_lines'])>0
                rows.append({'index':i,'lines':int(x['n_lines'])})
            return {'images':len(d),'smoke_only':ready['smoke_only'],'rendered_inputs':rows,'source_release':ready.get('source_release')}
        check('native_dataset_loader',dataset)
    if a.gpu:
        def gpu():
            import torch
            assert torch.cuda.is_available() and torch.cuda.device_count()>=8
            return {'cuda_runtime':torch.version.cuda,'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        check('gpu',gpu)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
