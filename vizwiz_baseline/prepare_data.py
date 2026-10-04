"""Only download test images; stream official combined CSVs and persist TEST rows only."""
import argparse, collections, csv, difflib, hashlib, io, json, subprocess, urllib.request, zipfile
from pathlib import Path
from protocol import DATA, LABELS
URL_BASE = 'https://vizwiz.cs.colorado.edu/'
QUALITY_URL = URL_BASE + 'VizWiz_all_answers/VizWiz_quality_issues_train_val_test.csv'
CAPTION_URL = URL_BASE + 'VizWiz_all_answers/VizWiz_captions_project_all_metadata.csv'
MANIFEST_URL = 'https://raw.githubusercontent.com/chiutaiyin/VizWiz-QualityIssues/master/annotations/quality_annotations/test.json'
SEVERE = 'quality issues are too severe to recognize visual content'
def build_mapping():
    sources = {
        'legacy_test_questions.json': 'https://vizwiz.cs.colorado.edu/VizWiz/data/Annotations/test.json',
        'current_test_questions.json': 'https://raw.githubusercontent.com/chiutaiyin/VizWiz-QualityIssues/master/annotations/vqa_annotations/test.json',
    }
    for name, url in sources.items():
        if not (DATA/name).exists(): (DATA/name).write_bytes(urllib.request.urlopen(url).read())
    old = sorted(json.loads((DATA/'legacy_test_questions.json').read_text()),key=lambda r:r['image'])
    new = sorted(json.loads((DATA/'current_test_questions.json').read_text()),key=lambda r:r['image'])
    matcher = difflib.SequenceMatcher(a=[r['question'] for r in old], b=[r['question'] for r in new], autojunk=False)
    pairs = [(i+d,j+d) for i,j,n in matcher.get_matching_blocks() for d in range(n)]
    assert len(pairs)==7974 and [j for i,j in pairs]==list(range(7974))
    # Prove the exact question sequence has a unique monotone alignment, including repeated generic questions.
    def align(a,b):
        result=[];i=0
        for r in b:
            while i<len(a) and a[i]['question']!=r['question']: i+=1
            assert i<len(a)
            result.append(i);i+=1
        return result
    lo=align(old,new[:7974])
    hi=[len(old)-1-i for i in align(old[::-1],new[:7974][::-1])][::-1]
    assert lo==hi==[i for i,j in pairs], 'Ambiguous legacy-to-current mapping'
    mapping={old[i]['image']:new[j]['image'] for i,j in pairs}
    audit={'method':'Unique monotone exact-question alignment of two official TEST question lists; earliest and latest alignments identical',
           'mapped':len(mapping),'deleted_legacy':[r['image'] for r in old if r['image'] not in mapping],
           'unmapped_current':[r['image'] for r in new if r['image'] not in mapping.values()],
           'mapping':mapping,'sources':sources,
           'sha256':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in sources}}
    (DATA/'filename_mapping.json').write_text(json.dumps(audit,indent=2))
    return mapping
def severe(caption):
    return caption.strip().lower().rstrip('.! ') == SEVERE

def stream_test(url, out, split_field, encoding, columns=None):
    if out.exists(): return
    with urllib.request.urlopen(url, timeout=120) as response, out.with_suffix('.partial').open('w') as f:
        reader = csv.DictReader(io.TextIOWrapper(response, encoding=encoding))
        columns = columns or reader.fieldnames
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for row in reader:
            if row[split_field] == 'TEST': writer.writerow({k:row[k] for k in columns})
    out.with_suffix('.partial').replace(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--download', action='store_true'); args=ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.download:
        archive=DATA/'test.zip'
        if not zipfile.is_zipfile(archive):
            subprocess.run(['curl','-fL','--retry','4','-C','-',URL_BASE+'VizWiz_final/images/test.zip','-o',str(archive)],check=True)
        with zipfile.ZipFile(archive) as z:
            assert all(n.startswith('test/') and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(DATA/'images')
    stream_test(QUALITY_URL,DATA/'quality_test_raw.csv','SPLIT','utf-8-sig')
    stream_test(CAPTION_URL,DATA/'captions_test_raw.csv','TRAIN_VAL_TEST','cp1252',
                ['IMG','WORKERID','IMG_CAPTION','IS_REJECTED','TRAIN_VAL_TEST'])
    mp=DATA/'test_manifest_official.json'
    if not mp.exists(): mp.write_bytes(urllib.request.urlopen(MANIFEST_URL).read())
    names=sorted(r['image'] for r in json.loads(mp.read_text()))
    assert len(names)==len(set(names))==8000
    assert set(names)=={f'VizWiz_test_{i:08d}.jpg' for i in range(8000)}
    manifest=[{'image':n,'path':str(DATA/'images/test'/n)} for n in names]
    assert all(Path(r['path']).is_file() for r in manifest)
    mapping=build_mapping()
    groups={}
    for kind in ['quality','captions']:
        rows=list(csv.DictReader((DATA/f'{kind}_test_raw.csv').open()))
        g=collections.defaultdict(list)
        for r in rows:
            if r['IMG'] in mapping: g[mapping[r['IMG']]].append(r)
        assert set(g)<=set(names)
        assert all(len(rs)==5 and len({r['WORKERID'] for r in rs})==5 for rs in g.values())
        groups[kind]=g
    inverse={v:k for k,v in mapping.items()}
    output=[]
    for n in names:
        entry={'image':n,'legacy_image':inverse.get(n)}
        for policy in ['accepted','all']:
            votes={}; counts={}
            q=[r for r in groups['quality'].get(n,[]) if policy=='all' or r['REJECT']=='0']
            c=[r for r in groups['captions'].get(n,[]) if policy=='all' or r['IS_REJECTED']=='0']
            for label in LABELS[1:]:
                votes[label]=sum(int(r[label]) for r in q) if q else None; counts[label]=len(q)
            votes['UNREC']=sum(severe(r['IMG_CAPTION']) for r in c) if c else None; counts['UNREC']=len(c)
            entry[policy]={'votes':votes,'workers':counts,
                'labels':{k:int(votes[k]>=2) if counts[k]>=2 else None for k in LABELS}}
        output.append(entry)
    (DATA/'test_manifest.json').write_text(json.dumps(manifest,indent=2))
    (DATA/'test_labels.json').write_text(json.dumps(output,indent=2))
    audit={'image_count':len(names),'quality_annotated_images':len(groups['quality']),
      'caption_annotated_images':len(groups['captions']),
      'quality_missing':[n for n in names if n not in groups['quality']],
      'caption_missing':[n for n in names if n not in groups['captions']],
      'sources':{'quality':QUALITY_URL,'captions':CAPTION_URL,'manifest':MANIFEST_URL},
      'filename_mapping':'Unique monotone exact-question alignment of official old/new TEST lists; 7974 matched, 26 new images unlinked. See filename_mapping.json. Validated with image/caption spot checks.',
      'label_rule':'At least two votes. Main: exclude rejected workers; sensitivity: retain all original five workers. Fewer than two usable workers -> missing, not negative.',
      'unrecognizable_rule':'Exact normalized caption match to: '+SEVERE,
      'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'quality_test_raw.csv',DATA/'captions_test_raw.csv',mp,DATA/'test_manifest.json',DATA/'test_labels.json']},
      'coverage':{policy:{k:sum(x[policy]['labels'][k] is not None for x in output) for k in LABELS} for policy in ['accepted','all']}}
    (DATA/'annotation_audit.json').write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
