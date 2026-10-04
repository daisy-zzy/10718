"""Strict full-test coverage checks, masked metrics, bootstrap CIs, and report artifacts."""
import argparse, hashlib, json, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/vizwiz_matplotlib')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score, precision_recall_curve, roc_curve,
    accuracy_score, hamming_loss, jaccard_score, brier_score_loss, log_loss)
from protocol import DATA, LABELS

SIX = ['BLR','BRT','DRK','OBS','FRM','ROT']

def load_predictions(paths, expected):
    rows={}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r=json.loads(line); name=r['image']
            if name in rows: raise ValueError(f'Duplicate prediction: {name}')
            if set(r['scores'])!=set(LABELS): raise ValueError('Unexpected label names')
            p=np.array([r['scores'][k] for k in LABELS])
            if not (np.isfinite(p).all() and ((p>=0)&(p<=1)).all()): raise ValueError('Invalid probabilities')
            rows[name]=r
    if set(rows)!=set(expected): raise ValueError(f'Prediction coverage mismatch: got {len(rows)}, expected {len(expected)}')
    return [rows[n] for n in expected]

def ece(y,p):
    ids=np.minimum((p*10).astype(int),9)
    return sum(np.mean(ids==b)*abs(p[ids==b].mean()-y[ids==b].mean()) for b in range(10) if (ids==b).any())

def binary_metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p);pred=(p>0.5).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(y),'positives':int(y.sum()),'prevalence':float(y.mean()),
        'AP':float(average_precision_score(y,p)) if y.sum() else None,
        'AUROC':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        'precision':float(precision),'recall':float(recall),'F1':float(f1),
        'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'specificity':float(tn/(tn+fp)) if tn+fp else None,'MCC':float(matthews_corrcoef(y,pred)),
        'Brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
        'ECE_10bins':float(ece(y,p)),'predicted_positive_rate':float(pred.mean()),
        'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def multilabel(y,p):
    pred=p>0.5
    out={'n':len(y),'mAP':float(average_precision_score(y,p,average='macro')),
         'micro_AP':float(average_precision_score(y,p,average='micro')),
         'exact_match':float(accuracy_score(y,pred)),'hamming_loss':float(hamming_loss(y,pred)),
         'sample_Jaccard':float(jaccard_score(y,pred,average='samples',zero_division=0)),
         'label_cardinality_true':float(y.sum(1).mean()),'label_cardinality_pred':float(pred.sum(1).mean())}
    for avg in ['macro','micro','weighted','samples']:
        pr,re,f1,_=precision_recall_fscore_support(y,pred,average=avg,zero_division=0)
        out.update({f'{avg}_precision':float(pr),f'{avg}_recall':float(re),f'{avg}_F1':float(f1)})
    return out

class BootstrapMetrics:
    """Cache score ordering so tied-score AP is exactly sklearn AP under row weights."""
    def __init__(self,y,p):
        self.y=y;self.p=p;self.pred=p>0.5;self.cache=[]
        for j in range(y.shape[1]):
            order=np.argsort(-p[:,j],kind='stable')
            ends=np.r_[np.flatnonzero(np.diff(p[order,j])),len(y)-1]
            self.cache.append((order,ends))
    def compute(self,w):
        y=self.y;pred=self.pred;ww=w[:,None]
        tp=(ww*(y*pred)).sum(0);fp=(ww*((1-y)*pred)).sum(0);fn=(ww*(y*(1-pred))).sum(0)
        den=2*tp+fp+fn;f1=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)
        aps=[]
        for j,(order,ends) in enumerate(self.cache):
            weight=w[order];tps=np.cumsum(weight*y[order,j])[ends];total=np.cumsum(weight)[ends]
            prec=np.divide(tps,total,out=np.zeros_like(tps,dtype=float),where=total>0)
            aps.append(float(np.sum(np.diff(np.r_[0,tps])*prec)/tps[-1]) if tps[-1] else np.nan)
        return np.r_[aps,f1]

def bootstrap(y, scores, count=1000):
    rng=np.random.default_rng(2026);n=len(y);models=list(scores)
    engines={name:BootstrapMetrics(y,p) for name,p in scores.items()}
    collected={name:[] for name in models};width=len(LABELS)
    def summarize(v,p,w):
        aps=v[:width];f1=v[width:];pred=p[:,1:]>0.5;gt=y[:,1:];ww=w[:,None]
        tp=(ww*(gt*pred)).sum();fp=(ww*((1-gt)*pred)).sum();fn=(ww*(gt*(1-pred))).sum()
        return np.r_[v,aps[1:].mean(),f1[1:].mean(),2*tp/(2*tp+fp+fn),aps[[LABELS.index(k) for k in SIX]].mean()]
    metric_names=[k+'_AP' for k in LABELS]+[k+'_F1' for k in LABELS]+['quality8_mAP','quality8_macro_F1','quality8_micro_F1','quality6_mAP']
    for _ in range(count):
        w=np.bincount(rng.integers(n,size=n),minlength=n)
        for name in models: collected[name].append(summarize(engines[name].compute(w),scores[name],w))
    result={'method':'Paired image-level percentile bootstrap; identical resamples across models','resamples':count,'seed':2026,'n':n,'models':{}}
    for name in models:
        arr=np.asarray(collected[name]);lo,hi=np.nanpercentile(arr,[2.5,97.5],axis=0)
        result['models'][name]={k:{'low':float(l),'high':float(h)} for k,l,h in zip(metric_names,lo,hi)}
    if len(models)==2:
        delta=np.asarray(collected[models[1]])-np.asarray(collected[models[0]])
        lo,hi=np.nanpercentile(delta,[2.5,97.5],axis=0)
        result['difference']={'direction':models[1]+' minus '+models[0],
            'metrics':{k:{'low':float(l),'high':float(h)} for k,l,h in zip(metric_names,lo,hi)}}
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=Path('results/independent'))
    ap.add_argument('--output', type=Path, help='Evaluation output directory')
    ap.add_argument('--bootstrap', type=int, default=1000)
    args = ap.parse_args()
    out = args.output or args.results / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    labels = json.loads((DATA / 'test_labels.json').read_text())
    names = [r['image'] for r in labels]
    specs = {
        name: sorted((args.results / folder).glob('shard*.jsonl'))
        for name, folder in [('Qwen3.5-0.8B', 'qwen35_08b'), ('Qwen3.5-2B', 'qwen35_2b')]
    }
    preds={name:load_predictions(paths,names) for name,paths in specs.items()}
    scores={name:np.array([[r['scores'][k] for k in LABELS] for r in rows]) for name,rows in preds.items()}
    complete=np.array([all(r['accepted']['labels'][k] is not None for k in LABELS) for r in labels])
    y=np.array([[r['accepted']['labels'][k] for k in LABELS] for r in labels if all(r['accepted']['labels'][k] is not None for k in LABELS)])
    assert len(y)==7974, 'Unexpected annotation coverage; review audit'
    result={'protocol':'independent','prediction_images':len(names),'evaluation_images':len(y),'missing_images':np.array(names)[~complete].tolist(),
        'classification_threshold':0.5,'positive_rule':'score > 0.5; ties -> 0','ground_truth_vote_threshold':2,
        'label_order':LABELS,'models':{},'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for paths in specs.values() for p in paths}}
    rows=[];summaries=[];sensitivity=[];error_rows=[]
    for name,score in scores.items():
        p=score[complete];per={k:binary_metrics(y[:,j],p[:,j]) for j,k in enumerate(LABELS)}
        multi8=multilabel(y[:,1:],p[:,1:]);six_idx=[LABELS.index(k) for k in SIX];multi6=multilabel(y[:,six_idx],p[:,six_idx])
        binary_pred=p>0.5
        result['models'][name]={'per_label':per,'quality8':multi8,'quality6':multi6,
            'NON_and_specific_flaw_rate':float((binary_pred[:,1]&binary_pred[:,2:].any(1)).mean()),
            'binary_token_mass_mean':{k:float(np.mean([r['binary_token_mass'][k] for r in preds[name]])) for k in LABELS}}
        for k,m in per.items(): rows.append({'model':name,'label':k,**m})
        summary={'model':name,'n_predicted':len(names),'n_evaluated':len(y),'UNREC_AP':per['UNREC']['AP'],'UNREC_precision':per['UNREC']['precision'],
             'UNREC_recall':per['UNREC']['recall'],'UNREC_F1':per['UNREC']['F1'],'quality8_mAP':multi8['mAP'],'quality8_macro_F1':multi8['macro_F1'],
             'quality8_micro_F1':multi8['micro_F1'],'quality6_mAP':multi6['mAP'],'quality6_macro_F1':multi6['macro_F1'],'quality8_exact_match':multi8['exact_match']}
        summaries.append(summary)
        # Annotation-policy sensitivity only; no prompt or threshold selection on test labels.
        for policy,vote_threshold in [('all',2),('accepted',3)]:
            eligible=np.array([all(r[policy]['workers'][k]>=vote_threshold for k in LABELS) for r in labels]) & complete
            yy=np.array([[int(r[policy]['votes'][k]>=vote_threshold) for k in LABELS] for r,ok in zip(labels,eligible) if ok])
            pp=score[eligible]
            un=binary_metrics(yy[:,0],pp[:,0]);mu=multilabel(yy[:,1:],pp[:,1:])
            sensitivity.append({'model':name,'worker_policy':policy,'vote_threshold':vote_threshold,'n':int(eligible.sum()),'UNREC_AP':un['AP'],'UNREC_F1':un['F1'],
                                'quality8_mAP':mu['mAP'],'quality8_macro_F1':mu['macro_F1'],'quality8_micro_F1':mu['micro_F1']})
        for j,k in enumerate(LABELS):
            bad=np.flatnonzero(binary_pred[:,j]!=y[:,j]);order=bad[np.argsort(-np.abs(p[bad,j]-.5))][:20]
            for i in order: error_rows.append({'model':name,'image':np.array(names)[complete][i],'label':k,'truth':int(y[i,j]),'prediction':int(binary_pred[i,j]),'score':p[i,j]})
        # Joined per-image table is separate from the label-blind inference artifacts.
        joined=[]
        for i,n in enumerate(names):
            joined.append({'image':n,**{k+'_score':score[i,j] for j,k in enumerate(LABELS)},
                           **{k+'_truth':labels[i]['accepted']['labels'][k] for k in LABELS}})
        pd.DataFrame(joined).to_csv(out/(name+'_predictions.csv'),index=False)
    runtime=[]
    for name,paths in specs.items():
        records=[json.loads(p.with_suffix('.runtime.json').read_text()) for p in paths]
        runtime.append({'model':name,'GPU_workers':len(records),'images':sum(r['new_images'] for r in records),
            'inference_wall_seconds_approx':max(p.with_suffix('.runtime.json').stat().st_mtime for p in paths)-min(p.with_suffix('.config.json').stat().st_mtime for p in paths),
            'sum_worker_seconds':sum(r['wall_seconds'] for r in records),
            'peak_allocated_GiB_per_GPU':max(r['peak_gpu_allocated_bytes'] for r in records)/2**30,
            'images_per_GPU_second':sum(r['new_images'] for r in records)/sum(r['wall_seconds'] for r in records)})
    result['runtime']=runtime
    pd.DataFrame(runtime).to_csv(out/'runtime.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'per_class_metrics.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'annotation_sensitivity.csv',index=False)
    pd.DataFrame(error_rows).to_csv(out/'confident_errors.csv',index=False)
    result['annotation_sensitivity']=sensitivity
    if args.bootstrap:
        print('Bootstrap started',flush=True)
        result['bootstrap']=bootstrap(y,{name:p[complete] for name,p in scores.items()},args.bootstrap)
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    # Curves and confusion matrices for all nine labels.
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    for curve in ['PR','ROC']:
        fig,axs=plt.subplots(3,3,figsize=(13,11))
        for j,(k,ax) in enumerate(zip(LABELS,axs.flat)):
            data={}
            for name,score in scores.items():
                p=score[complete,j]
                if curve=='PR':
                    precision,recall,threshold=precision_recall_curve(y[:,j],p);xx,yy=recall,precision
                    data[name]={'recall':recall.tolist(),'precision':precision.tolist(),'thresholds':threshold.tolist()}
                else:
                    xx,yy,threshold=roc_curve(y[:,j],p);data[name]={'fpr':xx.tolist(),'tpr':yy.tolist(),'thresholds':[float(v) if np.isfinite(v) else None for v in threshold]}
                ax.plot(xx,yy,label=name)
            if curve=='PR':ax.axhline(y[:,j].mean(),color='gray',ls=':',label='Positive prevalence')
            else:ax.plot([0,1],[0,1],color='gray',ls=':')
            ax.set(title=k,xlim=(0,1),ylim=(0,1),xlabel='Recall' if curve=='PR' else 'False positive rate',ylabel='Precision' if curve=='PR' else 'True positive rate')
            if j==0:ax.legend(fontsize=7)
            (out/f'{curve}_{k}.json').write_text(json.dumps(data,allow_nan=False))
        fig.tight_layout();fig.savefig(out/(curve+'_curves.png'));fig.savefig(out/(curve+'_curves.pdf'));plt.close(fig)
    for name in scores:
        fig,axs=plt.subplots(3,3,figsize=(11,10))
        for k,ax in zip(LABELS,axs.flat):
            m=result['models'][name]['per_label'][k];cm=np.array([[m['TN'],m['FP']],[m['FN'],m['TP']]])
            ax.imshow(cm,cmap='Blues');ax.set(title=k,xlabel='Predicted',ylabel='True',xticks=[0,1],yticks=[0,1])
            for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
        fig.suptitle(name+' (threshold 0.5)');fig.tight_layout();fig.savefig(out/(name+'_confusion.png'));plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    xx=np.arange(len(LABELS));width=.36
    for i,name in enumerate(scores):
        for ax,metric in zip(axs,['AP','F1']):
            ax.bar(xx+(i-.5)*width,[100*result['models'][name]['per_label'][k][metric] for k in LABELS],width,label=name)
            ax.set(xticks=xx,xticklabels=LABELS,ylabel=metric+' (%)',ylim=(0,100));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'per_class_comparison.png');fig.savefig(out/'per_class_comparison.pdf');plt.close(fig)
    # Fixed threshold sweep is diagnostic only; no test-selected optimum is reported.
    ts=[]
    for name,score in scores.items():
        for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
            for j,k in enumerate(LABELS):
                pr,re,f1,_=precision_recall_fscore_support(y[:,j],score[complete,j]>t,average='binary',zero_division=0)
                ts.append({'model':name,'threshold':t,'label':k,'precision':pr,'recall':re,'F1':f1})
    pd.DataFrame(ts).to_csv(out/'fixed_threshold_diagnostics.csv',index=False)
    table=pd.DataFrame(summaries);formatted=table.copy()
    for c in formatted:
        if c not in ['model','n_predicted','n_evaluated']:formatted[c]=formatted[c].map(lambda x:f'{100*x:.2f}')
    formatted.to_latex(out/'summary.tex',index=False,escape=True)
    report = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<title>VizWiz baseline results</title>',
        '<style>body{font-family:sans-serif;max-width:1200px;margin:40px auto}',
        'table{border-collapse:collapse;display:block;overflow:auto}',
        'th,td{border:1px solid #ddd;padding:6px}img{max-width:100%}</style>',
        '<h1>VizWiz baseline results</h1>',
        f'<p>{len(names)} predicted images; {len(y)} labeled images. Decision threshold: score &gt; 0.5.</p>',
        '<h2>Summary (%)</h2>', formatted.to_html(index=False),
        '<h2>Per-label metrics (0–1)</h2>', pd.DataFrame(rows).to_html(index=False),
        '<h2>Runtime</h2>', pd.DataFrame(runtime).to_html(index=False),
    ]
    if args.bootstrap:
        cirows = [
            {'model': name, 'metric': metric, **interval}
            for name, metrics in result['bootstrap']['models'].items()
            for metric, interval in metrics.items()
        ]
        report += [f'<h2>95% confidence intervals ({args.bootstrap} resamples)</h2>',
                   pd.DataFrame(cirows).to_html(index=False)]
    report += ['<h2>Annotation sensitivity (0–1)</h2>',
               pd.DataFrame(sensitivity).to_html(index=False), '<h2>Plots</h2>']
    for filename in ['per_class_comparison.png', 'PR_curves.png', 'ROC_curves.png'] + [
        name + '_confusion.png' for name in scores
    ]:
        report.append(f'<img src="{filename}" alt="{filename}">')
    report.append('</html>')
    (out / 'report.html').write_text('\n'.join(report))
    print(json.dumps(summaries, indent=2))
    print('Report:', out / 'report.html')

if __name__ == '__main__':
    main()
