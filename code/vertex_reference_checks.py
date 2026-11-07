"""Validate vertex partitions on archived inputs and every edge of the large ladders."""
import argparse
import csv
import json
import math
import random
from decimal import Decimal, localcontext
from pathlib import Path

from vertex_outage import all_outages_vertex, compiled, inputs
from vertex_study import decimal_solve, ladder, write_csv, write_json

RESULTS = Path(__file__).resolve().parents[1] / "results"
RANGE = ["hw_underflow", "subnormal", "positive_to_zero", "outer_subnormal", "outer_zero", "nonfinite"]


def rel(value, reference):
    return float(abs(Decimal.from_float(float(value)) / reference - 1))


def other_suites():
    detailed = []
    summaries = []
    graphs_audit = []
    for suite, filename in [("accuracy", "accuracy_inputs.json"), ("pglib", "revision_pglib_inputs.json"),
                            ("rank", "revision_rank_inputs.json")]:
        graphs = json.loads((RESULTS / filename).read_text())
        with (RESULTS / f"joint_{suite}.csv").open() as source:
            refs = {(row["case"], int(row["eid"])): row for row in csv.DictReader(source)}
        for g in graphs:
            case = g["case"]
            p, audit = all_outages_vertex(g["n"], g["edges"], strict_range=False)
            outside = any(audit[key] for key in RANGE)
            graphs_audit.append(dict(suite=suite, case=case, outside_range=outside, **audit))
            for e in range(len(g["edges"])):
                if e in p.bridges:
                    assert (case,e) not in refs and math.isinf(p.bypass[e]) and p.defect[e] == 0
                    continue
                ref = refs.pop((case,e))
                E = p.error_exponents[e]
                er = rel(p.bypass[e], Decimal(ref["rho_reference"]))
                ed = rel(p.defect[e], Decimal(ref["delta_reference"]))
                br, bd = [math.expm1(-t*math.log1p(-2**-53)) for t in (E,E+2)]
                detailed.append(dict(suite=suite,case=case,edge=e,E=E,rho_err=er,delta_err=ed,
                                     rho_ratio=er/br,delta_ratio=ed/bd,outside_range=outside))
        assert not refs
        rows=[q for q in detailed if q["suite"]==suite]
        for sub in (["all","s0","s8","s16","s32"] if suite=="accuracy" else ["all"]):
            part=[q for q in rows if sub=="all" or "_"+sub+"_" in q["case"]]
            stat=dict(suite=suite, subset=sub, graphs=len({q["case"] for q in part}), queries=len(part),
                      outside_graphs=len({q["case"] for q in part if q["outside_range"]}),
                      outside_queries=sum(q["outside_range"] for q in part))
            for name in ["rho","delta"]:
                top=max(part,key=lambda q:q[name+"_ratio"])
                stat.update({name+"_max":max(q[name+"_err"] for q in part),name+"_max_ratio":top[name+"_ratio"],
                             name+"_case":top["case"],name+"_edge":top["edge"],
                             name+"_violations":sum(q[name+"_ratio"]>1 for q in part)})
            summaries.append(stat)
    write_csv("vertex_archived_queries.csv", detailed)
    write_csv("vertex_archived_range.csv", graphs_audit)
    write_json("vertex_archived_summary.json", summaries)
    for s in summaries: print(s,flush=True)


def all_ladders():
    summary=[]
    detail=[]
    for n in (128,256,384):
        for spread in (8,32):
            edges=ladder(n,spread,random.Random(48600+n*37+spread))
            ar=inputs(edges); rho, audit=compiled(n,ar,3)
            delta=ar[2]/(ar[2]+rho)
            order=sorted(range(len(edges)),key=lambda e:delta[e])
            errors=[]
            for rank,e in enumerate(order,1):
                s,_,r=edges[e]
                x130=decimal_solve(n,edges,e,130)[s]
                x210=decimal_solve(n,edges,e,210)[s]
                r=Decimal.from_float(r); ref_delta=r/(r+x210)
                record=dict(v=n,spread=spread,e=e,delta_rank=rank,rho_ref=str(x210),delta_ref=str(ref_delta),
                            rho_error=rel(rho[e],x210),delta_error=rel(delta[e],ref_delta),
                            precision_gap=float(abs(x130/x210-1)))
                errors.append(record)
            detail+=errors
            summary.append(dict(v=n,spread=spread,queries=len(errors),rho_error=max(e["rho_error"] for e in errors),
                                delta_error=max(e["delta_error"] for e in errors),
                                precision_gap=max(e["precision_gap"] for e in errors),**audit))
            write_csv("vertex_all_ladder_references.csv",detail)
            write_json("vertex_all_ladders.json",summary)
            print(summary[-1],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("study",choices=["suites","ladders"])
    with localcontext() as context:
        context.prec=225
        (other_suites if parser.parse_args().study=="suites" else all_ladders)()
