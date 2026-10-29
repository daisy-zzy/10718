"""Independent one-token Yes/No scoring; no label chaining or output parsing."""
import argparse,hashlib,json,time,platform
from pathlib import Path
import torch,transformers
from PIL import Image
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from protocol import ROOT,DATA,LABELS
QUESTIONS={
'UNREC':'Are the image quality problems so severe that the content cannot be recognized well enough to describe what is shown?',
'NON':'Is this image free of photographic quality flaws such as blur, overexposure, underexposure, obstructions, poor framing, and rotation?',
'BLR':'Is this image blurry, out of focus, or affected by motion blur?',
'BRT':'Is this image too bright or overexposed, obscuring visual details?',
'DRK':'Is this image too dark or underexposed, obscuring visual details?',
'OBS':"Does a finger or another unintended obstruction block the camera's view in this image?",
'FRM':'Does improper framing cut off the main subject or fail to show it adequately in this image?',
'ROT':'Is this image rotated from its proper viewing orientation?',
'OTH':'Does this image have another photographic quality flaw besides blur, overexposure, underexposure, obstructions, improper framing, and rotation?'}
PROMPTS={k:'Assess the photographic quality of this image. '+q+' Answer only Yes or No.' for k,q in QUESTIONS.items()}
def answer_token_groups(tokenizer):
    groups=[]
    for word in ['no','yes']:
        variants=[prefix+w for prefix in ['', ' ', '\n'] for w in [word,word.title(),word.upper()]]
        ids={encoded[0] for text in variants if len(encoded:=tokenizer.encode(text,add_special_tokens=False))==1}
        groups.append(sorted(ids))
    assert groups[0] and groups[1] and not set(groups[0])&set(groups[1])
    return groups

def shared_vision_logits(model, inputs):
    # Identical images appear in nine independent language sequences. Encode each image once.
    grids=inputs['image_grid_thw'];n=len(LABELS)
    assert len(grids)%n==0
    assert torch.equal(grids,grids[::n].repeat_interleave(n,dim=0))
    chunks=inputs['pixel_values'].split(grids.prod(-1).tolist())
    pixels=torch.cat(chunks[::n],dim=0)
    features=model.get_image_features(pixels,grids[::n],return_dict=True).pooler_output
    repeated=torch.cat([f for f in features for _ in LABELS],dim=0)
    embeds=model.get_input_embeddings()(inputs['input_ids'])
    mask,_=model.model.get_placeholder_mask(inputs['input_ids'],inputs_embeds=embeds,image_features=repeated)
    embeds=embeds.masked_scatter(mask,repeated.to(embeds.dtype))
    positions,_=model.model.get_rope_index(inputs['input_ids'],image_grid_thw=grids,
        attention_mask=inputs['attention_mask'],mm_token_type_ids=inputs['mm_token_type_ids'])
    return model(inputs_embeds=embeds,position_ids=positions,attention_mask=inputs['attention_mask'],
        logits_to_keep=1,use_cache=False).logits[:,-1].float()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--output',required=True,type=Path)
 ap.add_argument('--batch-size',type=int,default=144);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--num-shards',type=int,default=1);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--verify-shared-vision',action='store_true')
 args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(2026)
 manifest=DATA/'test_manifest.json';rows=json.loads(manifest.read_text())[args.shard::args.num_shards]
 if args.limit:rows=rows[:args.limit]
 path=ROOT/'models'/args.model;args.output.parent.mkdir(parents=True,exist_ok=True)
 config={'model':args.model,'model_provenance':json.loads((path/'download_provenance.json').read_text()),'labels':LABELS,'prompts':PROMPTS,
 'protocol':'independent_yes_no_v2','vision_reuse':True,'answer_variants':'single-token No/no/NO/space variants and Yes/yes/YES/space variants','dtype':'float16','attention':'sdpa','thinking':False,'max_pixels':262144,'min_pixels':4096,
 'batch_size':args.batch_size,'shard':args.shard,'num_shards':args.num_shards,'limit':args.limit,'seed':2026,
 'score':'sum probability of Yes variants / sum probability of Yes and No variants; each label independently conditioned on image and its question',
 'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),'python':platform.python_version(),
 'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 meta=args.output.with_suffix('.config.json')
 if meta.exists() and json.loads(meta.read_text())!=config:raise ValueError('Configuration changed')
 meta.write_text(json.dumps(config,indent=2))
 done={}
 if args.output.exists():
  for line in args.output.read_text().splitlines():
   r=json.loads(line)
   if r['image'] in done:raise ValueError('Duplicate')
   done[r['image']]=r
 todo=[r for r in rows if r['image'] not in done]
 if not todo:print('Already complete');return
 processor=AutoProcessor.from_pretrained(path,local_files_only=True);processor.tokenizer.padding_side='left'
 processor.image_processor.size={'shortest_edge':4096,'longest_edge':262144}
 model=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
 groups=answer_token_groups(processor.tokenizer)
 texts={k:processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPTS[k]}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for k in LABELS}
 # Process one image group at a time; each image produces 9 independent sequences.
 image_batch=max(1,args.batch_size//len(LABELS));timings=[];start=time.perf_counter()
 with torch.inference_mode(),args.output.open('a',buffering=1) as out:
  for offset in range(0,len(todo),image_batch):
   batch=todo[offset:offset+image_batch];tick=time.perf_counter();images=[];text=[]
   for r in batch:
    with Image.open(r['path']) as im:image=im.convert('RGB')
    images.extend([image]*len(LABELS));text.extend(texts[k] for k in LABELS)
   inputs=processor(text=text,images=images,padding=True,return_tensors='pt').to('cuda')
   for k,v in inputs.items():
    if v.is_floating_point():inputs[k]=v.half()
   logits=shared_vision_logits(model,inputs)
   if args.verify_shared_vision and offset==0:
    reference=model(**inputs,logits_to_keep=1,use_cache=False).logits[:,-1].float()
    ids=groups[0]+groups[1]
    delta=(logits[:,ids]-reference[:,ids]).abs()
    comparison={'max_abs_candidate_logit_error':delta.max().item(),'mean_abs_candidate_logit_error':delta.mean().item()}
    assert delta.max()<.15,comparison
    args.output.with_suffix('.vision_check.json').write_text(json.dumps(comparison,indent=2))
    print('Vision reuse check',comparison,flush=True)
    del reference
   pair=torch.stack([logits[:,g].logsumexp(-1) for g in groups],dim=-1)
   if not torch.isfinite(pair).all():raise RuntimeError('Nonfinite logits')
   probs=pair.softmax(-1)[:,1].reshape(-1,len(LABELS)).cpu().numpy()
   mass=(pair.logsumexp(-1)-logits.logsumexp(-1)).exp().reshape(-1,len(LABELS)).cpu().numpy()
   top=logits.argmax(-1).reshape(-1,len(LABELS)).cpu().tolist();torch.cuda.synchronize()
   elapsed=time.perf_counter()-tick;timings.append(elapsed)
   for i,r in enumerate(batch):
    ps=probs[i].tolist();bits=[int(v>.5) for v in ps]
    out.write(json.dumps({'image':r['image'],'scores':dict(zip(LABELS,ps)),'predictions':dict(zip(LABELS,bits)),
      'raw':','.join(map(str,bits)),'binary_token_mass':dict(zip(LABELS,mass[i].tolist())),
      'unconstrained_top_tokens':dict(zip(LABELS,processor.tokenizer.convert_ids_to_tokens(top[i]))),
      'batch_seconds':elapsed,'batch_size':len(batch)},allow_nan=False)+'\n')
   completed=min(offset+len(batch),len(todo));wall=time.perf_counter()-start
   if (offset//image_batch)%10==0 or completed==len(todo):print(json.dumps({'model':args.model,'completed':completed+len(done),'target':len(rows),'images_per_second':round(completed/wall,3),'eta_seconds':round((len(todo)-completed)*wall/completed),'peak_gpu_gib':round(torch.cuda.max_memory_allocated()/2**30,2)}),flush=True)
   del inputs,logits,images
 args.output.with_suffix('.runtime.json').write_text(json.dumps({'new_images':len(todo),'total_images':len(rows),'wall_seconds':time.perf_counter()-start,
 'batch_seconds':timings,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()},indent=2))
if __name__=='__main__':main()
