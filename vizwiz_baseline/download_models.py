"""Download pinned official model snapshots, keeping all weights under ROOT."""
import os,json
from concurrent.futures import ThreadPoolExecutor
from protocol import ROOT
os.environ.setdefault('HF_HOME',str(ROOT/'cache/huggingface'))
from huggingface_hub import snapshot_download
MODELS={'Qwen3.5-0.8B':'2fc06364715b967f1860aea9cf38778875588b17',
        'Qwen3.5-2B':'15852e8c16360a2fea060d615a32b45270f8a8fc'}
def download(name):
    path=ROOT/'models'/name;revision=MODELS[name]
    snapshot_download('Qwen/'+name,revision=revision,local_dir=path,
        allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','README.md','LICENSE*'],max_workers=4)
    (path/'download_provenance.json').write_text(json.dumps({'repo':'Qwen/'+name,'revision':revision},indent=2))
if __name__=='__main__':
    with ThreadPoolExecutor(2) as pool:list(pool.map(download,MODELS))
