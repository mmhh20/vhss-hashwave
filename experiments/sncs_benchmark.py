from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scipy.stats import wilcoxon

from hashwave.baselines import BaselineConfig, BeamMutationSearch, SimulatedAnnealingSearch, StandardGeneticProgramming
from hashwave.expr import size
from hashwave.search import HashEvolutionSearch, RandomProgramSearch, SearchConfig, Task


def make_task(name, fn, train, validation, test, binary=("add","sub","mul"), unary=("neg",)):
    return Task.from_function(name, fn, train_xs=train, validation_xs=validation, test_xs=test, allowed_binary=binary, allowed_unary=unary)


def task_catalog() -> dict[str, Task]:
    wide = tuple(x for x in range(-24, 25) if x not in range(-8, 9))
    small_wide = tuple(x for x in range(-16, 17) if x not in range(-6, 7))
    bit_test = tuple(range(24, 64))
    return {
        "lin_2x_p1": make_task("2*x+1", lambda x:2*x+1, range(-5,6),(-8,-7,7,8),wide),
        "lin_m3x_p4": make_task("-3*x+4", lambda x:-3*x+4, range(-5,6),(-8,-7,7,8),wide),
        "lin_5x_m2": make_task("5*x-2", lambda x:5*x-2, range(-5,6),(-8,-7,7,8),wide),
        "lin_x_m7": make_task("x-7", lambda x:x-7, range(-5,6),(-8,-7,7,8),wide),
        "lin_m2x_m5": make_task("-2*x-5", lambda x:-2*x-5, range(-5,6),(-8,-7,7,8),wide),
        "quad_x2_x_1": make_task("x^2+x+1", lambda x:x*x+x+1, range(-6,7),(-10,-8,8,10),wide),
        "quad_2x2_m3x_2": make_task("2*x^2-3*x+2", lambda x:2*x*x-3*x+2, range(-6,7),(-10,-8,8,10),wide),
        "quad_3x2_2x_1": make_task("3*x^2+2*x+1", lambda x:3*x*x+2*x+1, range(-6,7),(-10,-8,8,10),wide),
        "quad_x2_m4": make_task("x^2-4", lambda x:x*x-4, range(-6,7),(-10,-8,8,10),wide),
        "quad_mx2_4x_3": make_task("-x^2+4*x+3", lambda x:-x*x+4*x+3, range(-6,7),(-10,-8,8,10),wide),
        "cubic_x3_x": make_task("x^3+x", lambda x:x*x*x+x, range(-5,6),(-8,-7,7,8),small_wide),
        "cubic_x3_m2x_4": make_task("x^3-2*x+4", lambda x:x*x*x-2*x+4, range(-5,6),(-8,-7,7,8),small_wide),
        "cubic_2x3_mx2_1": make_task("2*x^3-x^2+1", lambda x:2*x*x*x-x*x+1, range(-5,6),(-8,-7,7,8),small_wide),
        "cubic_x3_m3x2_2": make_task("x^3-3*x^2+2", lambda x:x*x*x-3*x*x+2, range(-5,6),(-8,-7,7,8),small_wide),
        "cubic_mx3_2x": make_task("-x^3+2*x", lambda x:-x*x*x+2*x, range(-5,6),(-8,-7,7,8),small_wide),
        "quartic_x4_x2_1": make_task("x^4+x^2+1", lambda x:x**4+x*x+1, range(-4,5),(-7,-6,6,7),small_wide),
        "quartic_2x4_mx2_3": make_task("2*x^4-x^2+3", lambda x:2*x**4-x*x+3, range(-4,5),(-7,-6,6,7),small_wide),
        "quartic_square": make_task("x^4-4*x^2+4", lambda x:x**4-4*x*x+4, range(-4,5),(-7,-6,6,7),small_wide),
        "abs_x": make_task("abs(x)", abs, range(-7,8),(-12,-9,9,12),wide, unary=("neg","abs")),
        "abs_2x_m3_p1": make_task("abs(2*x-3)+1", lambda x:abs(2*x-3)+1, range(-7,8),(-12,-9,9,12),wide, unary=("neg","abs")),
        "abs_double": make_task("abs(x+2)+abs(x-2)", lambda x:abs(x+2)+abs(x-2), range(-7,8),(-12,-9,9,12),wide, unary=("neg","abs")),
        "max_x_0": make_task("max(x,0)", lambda x:max(x,0), range(-8,9),(-12,-10,10,12),wide, binary=("add","sub","min","max")),
        "clamp_3": make_task("min(max(x,-3),3)", lambda x:min(max(x,-3),3), range(-8,9),(-12,-10,10,12),wide, binary=("add","sub","min","max")),
        "maxmin_gap": make_task("max(x,2)-min(x,-2)", lambda x:max(x,2)-min(x,-2), range(-8,9),(-12,-10,10,12),wide, binary=("add","sub","min","max")),
        "max_two_lines": make_task("max(2*x+1,-x+3)", lambda x:max(2*x+1,-x+3), range(-8,9),(-12,-10,10,12),wide, binary=("add","sub","mul","min","max")),
        "bit_xor3": make_task("x XOR 3", lambda x:x^3, range(0,16),range(16,24),bit_test, binary=("add","sub","xor","and","or")),
        "bit_and_or": make_task("(x AND 3)+(x OR 1)", lambda x:(x&3)+(x|1), range(0,16),range(16,24),bit_test, binary=("add","sub","xor","and","or")),
        "bit_mix5": make_task("(x XOR 5)+(x AND 1)", lambda x:(x^5)+(x&1), range(0,16),range(16,24),bit_test, binary=("add","sub","xor","and","or")),
        "bit_or_and": make_task("(x OR 2)-(x AND 1)", lambda x:(x|2)-(x&1), range(0,16),range(16,24),bit_test, binary=("add","sub","xor","and","or")),
        "bit_mask": make_task("(x XOR 3) AND 7", lambda x:(x^3)&7, range(0,16),range(16,24),bit_test, binary=("add","sub","xor","and","or")),
    }


def as_record(result):
    return {
        "program": result.best.source,
        "train_mae": result.best.train_mae,
        "validation_mae": result.best.validation_mae,
        "test_mae": result.test_mae,
        "proposals": result.proposals,
        "unique_evaluations": result.unique_evaluations,
        "cache_hits": result.cache_hits,
        "elapsed_seconds": result.elapsed_seconds,
        "size": size(result.best.expr),
        "stopped_reason": result.stopped_reason,
    }


def run_job(job):
    task_name, seed_index, evaluations, selected_methods = job
    task = task_catalog()[task_name]
    seed = f"sncs-{task_name}-{seed_index}"
    # The evolutionary VHSS budget is set by generations * beam * candidates per parent.
    generations, beam, per_parent = 24, 14, 28
    common = dict(generations=generations, beam_width=beam, candidates_per_parent=per_parent,
                  max_nodes=29 if task_name.startswith(("cubic","quartic")) else 25,
                  max_depth=10, patience=generations, seed=seed,
                  complexity_penalty=0.001, validation_weight=0.35,
                  semantic_support_weight=0.0)
    results = {}
    h = p = None
    if "vhss_hash" in selected_methods:
        h = HashEvolutionSearch(task, SearchConfig(**common, source="hash")).run()
        results["vhss_hash"] = as_record(h)
    if "vhss_prng" in selected_methods:
        p = HashEvolutionSearch(task, SearchConfig(**common, source="prng")).run()
        results["vhss_prng"] = as_record(p)
    observed = [r.proposals for r in (h, p) if r is not None]
    budget = max([evaluations] + observed)
    base = BaselineConfig(evaluations=budget, population_size=56, tournament_size=4,
                          max_nodes=common["max_nodes"], max_depth=10, seed=seed)
    if "standard_gp" in selected_methods:
        results["standard_gp"] = as_record(StandardGeneticProgramming(task, base).run())
    if "beam_mutation" in selected_methods:
        results["beam_mutation"] = as_record(BeamMutationSearch(task, base, beam_width=28).run())
    if "simulated_annealing" in selected_methods:
        results["simulated_annealing"] = as_record(SimulatedAnnealingSearch(task, base).run())
    if "random_program" in selected_methods:
        results["random_program"] = as_record(RandomProgramSearch(task, budget, seed=seed, max_nodes=common["max_nodes"]).run())
    return {"task":task_name,"seed":seed_index,"methods":results}


def rank_key(r):
    return (r["test_mae"] if r["test_mae"] is not None else float("inf"), r["validation_mae"], r["train_mae"], r["proposals"], r["size"])


def holm_adjust(pairs):
    ordered=sorted(enumerate(pairs), key=lambda x:x[1])
    out=[1.0]*len(pairs); running=0.0; m=len(pairs)
    for rank,(idx,p) in enumerate(ordered):
        adjusted=min(1.0,(m-rank)*p)
        running=max(running,adjusted); out[idx]=running
    return out


def summarize(rows):
    methods=sorted(set().union(*(r["methods"].keys() for r in rows)))
    summary={"methods":{},"families":{},"pairwise_vs_vhss_hash":{}}
    for method in methods:
        vals=[r["methods"][method] for r in rows if method in r["methods"]]
        summary["methods"][method]={
            "runs":len(vals),"test_exact":sum(v["test_mae"]==0 for v in vals),
            "test_exact_rate":sum(v["test_mae"]==0 for v in vals)/len(vals),
            "median_test_mae":statistics.median(v["test_mae"] for v in vals),
            "median_proposals":statistics.median(v["proposals"] for v in vals),
            "median_elapsed_seconds":statistics.median(v["elapsed_seconds"] for v in vals),
            "median_size":statistics.median(v["size"] for v in vals),
        }
    families={r["task"].split("_")[0] for r in rows}
    for family in sorted(families):
        fr=[r for r in rows if r["task"].startswith(family+"_")]
        summary["families"][family]={m:sum(r["methods"][m]["test_mae"]==0 for r in fr if m in r["methods"])/max(1,sum(m in r["methods"] for r in fr)) for m in methods}
    raw_ps=[]; names=[]
    if "vhss_hash" in methods:
        comparable=[r for r in rows if "vhss_hash" in r["methods"]]
        for method in methods:
            if method=="vhss_hash": continue
            paired=[r for r in comparable if method in r["methods"]]
            if not paired:
                continue
            a=[r["methods"]["vhss_hash"]["test_mae"] for r in paired]
            b=[r["methods"][method]["test_mae"] for r in paired]
            diffs=[x-y for x,y in zip(a,b)]
            nonzero=sum(d!=0 for d in diffs)
            try:
                p=float(wilcoxon(a,b,zero_method="pratt",alternative="two-sided").pvalue) if nonzero else 1.0
            except ValueError:
                p=1.0
            wins=sum(rank_key(r["methods"]["vhss_hash"])<rank_key(r["methods"][method]) for r in paired)
            losses=sum(rank_key(r["methods"]["vhss_hash"])>rank_key(r["methods"][method]) for r in paired)
            ties=len(paired)-wins-losses
            names.append(method); raw_ps.append(p)
            summary["pairwise_vs_vhss_hash"][method]={"paired_runs":len(paired),"vhss_wins":wins,"ties":ties,"vhss_losses":losses,"wilcoxon_p_raw":p}
        adjusted=holm_adjust(raw_ps)
        for name,p in zip(names,adjusted): summary["pairwise_vs_vhss_hash"][name]["holm_adjusted_p"]=p
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=20)
    ap.add_argument("--seed-start",type=int,default=0)
    ap.add_argument("--workers",type=int,default=max(1,min(8,os.cpu_count() or 1)))
    ap.add_argument("--evaluations",type=int,default=9000)
    ap.add_argument("--methods",default="vhss_hash,vhss_prng,standard_gp,beam_mutation,simulated_annealing,random_program")
    ap.add_argument("--output",default="results/sncs/benchmark.json")
    a=ap.parse_args()
    selected=tuple(x.strip() for x in a.methods.split(",") if x.strip())
    valid={"vhss_hash","vhss_prng","standard_gp","beam_mutation","simulated_annealing","random_program"}
    if not selected or set(selected)-valid: raise SystemExit(f"invalid methods: {sorted(set(selected)-valid)}")
    jobs=[(t,s,a.evaluations,selected) for t in task_catalog() for s in range(a.seed_start,a.seed_start+a.seeds)]
    started=time.perf_counter(); rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(run_job,j) for j in jobs]
        for i,f in enumerate(as_completed(fs),1):
            rows.append(f.result())
            if i%max(1,len(fs)//20)==0: print(f"completed {i}/{len(fs)}",flush=True)
    rows=sorted(rows,key=lambda r:(r["task"],r["seed"]))
    out={"configuration":{"tasks":len(task_catalog()),"seeds":a.seeds,"seed_start":a.seed_start,"workers":a.workers,"requested_evaluations":a.evaluations,"methods":selected},
         "summary":summarize(rows),"elapsed_seconds":time.perf_counter()-started,"runs":rows}
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out["summary"],indent=2))

if __name__=="__main__": main()
