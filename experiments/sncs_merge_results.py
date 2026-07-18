from __future__ import annotations
import argparse, json, math, random, statistics
from pathlib import Path
from scipy.stats import wilcoxon


def load(path): return json.loads(Path(path).read_text())
def keyrow(r): return (r['task'], int(r['seed']))
def rank_key(r): return (r['test_mae'] if r['test_mae'] is not None else float('inf'), r['validation_mae'], r['train_mae'], r['proposals'], r['size'])

def wilson(k,n,z=1.959963984540054):
    if n==0:return [0.0,0.0]
    p=k/n; den=1+z*z/n; centre=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,centre-half),min(1,centre+half)]

def sign_p(w,l):
    n=w+l
    if n==0:return 1.0
    k=min(w,l); tail=sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return min(1.0,2*tail)

def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]); out=[1.0]*len(ps); running=0
    for rank,i in enumerate(order):
        running=max(running,min(1.0,(len(ps)-rank)*ps[i])); out[i]=running
    return out

def summarize_method(rows,method):
    vals=[r['methods'][method] for r in rows if method in r['methods']]
    exact=sum(v['test_mae']==0 for v in vals); n=len(vals)
    return {'runs':n,'test_exact':exact,'test_exact_rate':exact/n,'test_exact_wilson95':wilson(exact,n),
            'median_test_mae':statistics.median(v['test_mae'] for v in vals),
            'median_proposals':statistics.median(v['proposals'] for v in vals),
            'median_elapsed_seconds':statistics.median(v['elapsed_seconds'] for v in vals),
            'median_program_size':statistics.median(v['size'] for v in vals)}

def paired(rows,a,b):
    rs=[r for r in rows if a in r['methods'] and b in r['methods']]
    w=sum(rank_key(r['methods'][a])<rank_key(r['methods'][b]) for r in rs)
    l=sum(rank_key(r['methods'][a])>rank_key(r['methods'][b]) for r in rs)
    t=len(rs)-w-l
    xa=[r['methods'][a]['test_mae'] for r in rs]; xb=[r['methods'][b]['test_mae'] for r in rs]
    try: wp=float(wilcoxon(xa,xb,zero_method='pratt').pvalue) if any(x!=y for x,y in zip(xa,xb)) else 1.0
    except ValueError: wp=1.0
    return {'paired_runs':len(rs),'a_wins':w,'ties':t,'a_losses':l,'sign_test_p':sign_p(w,l),'wilcoxon_test_mae_p':wp}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/sncs/merged_results.json'); a=ap.parse_args()
    base=Path(__file__).resolve().parents[1]/'results'/'sncs'
    primary=[]
    for f in ['primary_00_04.json','primary_05_09.json']:
        primary += load(base/f)['runs']
    primary=sorted({keyrow(r):r for r in primary}.values(),key=keyrow)
    # Baseline seed 0 comes from smoke; seeds 1 and 2 from dedicated runs.
    baseline_sources=[load(base/'smoke_30tasks_1seed.json')['runs'],load(base/'baselines_seed1.json')['runs'],load(base/'baselines_seed2.json')['runs']]
    basemap={}
    for source in baseline_sources:
        for r in source:
            k=keyrow(r); d=basemap.setdefault(k,{'task':r['task'],'seed':r['seed'],'methods':{}})
            for m,v in r['methods'].items():
                if m in {'standard_gp','beam_mutation','simulated_annealing','random_program'}: d['methods'][m]=v
    pmap={keyrow(r):r for r in primary}
    paired_baselines=[]
    for k,d in basemap.items():
        if k in pmap:
            merged={'task':d['task'],'seed':d['seed'],'methods':dict(d['methods'])}
            merged['methods'].update(pmap[k]['methods'])
            paired_baselines.append(merged)
    methods=['vhss_hash','vhss_prng','standard_gp','beam_mutation','simulated_annealing','random_program']
    summary={'primary_300_paired':{'methods':{m:summarize_method(primary,m) for m in ['vhss_hash','vhss_prng']},
                                   'vhss_hash_vs_prng':paired(primary,'vhss_hash','vhss_prng')},
             'baseline_90_paired':{'methods':{m:summarize_method(paired_baselines,m) for m in methods},'comparisons':{}},
             'task_count':30,'primary_seed_count':10,'baseline_seed_count':3}
    ps=[]; names=[]
    for m in methods[1:]:
        comp=paired(paired_baselines,'vhss_hash',m); summary['baseline_90_paired']['comparisons'][m]=comp; ps.append(comp['wilcoxon_test_mae_p']); names.append(m)
    for m,padj in zip(names,holm(ps)): summary['baseline_90_paired']['comparisons'][m]['holm_adjusted_wilcoxon_p']=padj
    output={'summary':summary,'primary_runs':primary,'baseline_paired_runs':sorted(paired_baselines,key=keyrow)}
    Path(a.output).write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
