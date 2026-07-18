from __future__ import annotations
import argparse, json, math, os, statistics, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from hashwave.search import HashEvolutionSearch, SearchConfig
from full_validation import task_catalog, comparison_key, sign_test_two_sided, result_dict

def run(job):
    task_name, seed_index, generations, beam, per_parent = job
    task=task_catalog()[task_name]
    seed=f'paired-{task_name}-{seed_index}'
    common=dict(generations=generations, beam_width=beam, candidates_per_parent=per_parent, patience=generations, seed=seed)
    h=HashEvolutionSearch(task, SearchConfig(**common, source='hash')).run()
    p=HashEvolutionSearch(task, SearchConfig(**common, source='prng')).run()
    return {'task':task_name,'seed':seed_index,'hash':result_dict(h),'prng':result_dict(p)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=20); ap.add_argument('--tasks',default='quadratic,cubic,absolute,bitmix'); ap.add_argument('--generations',type=int,default=40); ap.add_argument('--beam',type=int,default=18); ap.add_argument('--per-parent',type=int,default=40); ap.add_argument('--workers',type=int,default=max(1,min(8,os.cpu_count() or 1))); ap.add_argument('--output',default='results/hash_vs_prng_20x4.json'); a=ap.parse_args()
    tasks=[x.strip() for x in a.tasks.split(',') if x.strip()]; jobs=[(t,s,a.generations,a.beam,a.per_parent) for t in tasks for s in range(a.seeds)]
    started=time.perf_counter(); rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(run,j) for j in jobs]
        for i,f in enumerate(as_completed(fs),1):
            rows.append(f.result())
            if i%max(1,len(fs)//10)==0: print(f'completed {i}/{len(fs)}',flush=True)
    summary={}
    for task in tasks:
        rs=[r for r in rows if r['task']==task]; w=t=l=0
        for r in rs:
            ah=comparison_key(r['hash']); bp=comparison_key(r['prng'])
            if ah<bp:w+=1
            elif ah>bp:l+=1
            else:t+=1
        summary[task]={'runs':len(rs),'hash_test_exact':sum(r['hash']['test_mae']==0 for r in rs),'prng_test_exact':sum(r['prng']['test_mae']==0 for r in rs),'hash_wins':w,'ties':t,'hash_losses':l,'p':sign_test_two_sided(w,l),'hash_median_proposals':statistics.median(r['hash']['proposals'] for r in rs),'prng_median_proposals':statistics.median(r['prng']['proposals'] for r in rs)}
    w=t=l=0
    for r in rows:
        ah=comparison_key(r['hash']); bp=comparison_key(r['prng'])
        if ah<bp:w+=1
        elif ah>bp:l+=1
        else:t+=1
    summary['overall']={'runs':len(rows),'hash_wins':w,'ties':t,'hash_losses':l,'p':sign_test_two_sided(w,l)}
    out={'configuration':vars(a),'summary':summary,'elapsed_seconds':time.perf_counter()-started,'runs':sorted(rows,key=lambda r:(r['task'],r['seed']))}
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
