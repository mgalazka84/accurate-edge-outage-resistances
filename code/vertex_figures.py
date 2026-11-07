"""Render the matched four-method work and runtime comparison from recorded observations."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
STYLES = [("edge", "Edge partition (2)", "#235476", "o", "-"),
          ("vertex", "Vertex partition (3)", "#a03c2a", "s", "--"),
          ("sfldl", "Separate SF-LDL", "#4b6746", "^", "-."),
          ("inverse", "Grounded inverse", "#70558b", "D", ":")]


def render(output):
    with (ROOT / "results/vertex_matched_families.csv").open() as source:
        records = list(csv.DictReader(source))
    with plt.rc_context({"font.size": 9.2,"pdf.fonttype":42,"axes.linewidth":0.7}):
        fig,axes=plt.subplots(2,2,figsize=(6.8,4.4),sharex="col",sharey="row")
        for column,(family,title) in enumerate([("complete","Complete kernels"),("ladder","Circular ladders")]):
            for method,name,color,marker,style in STYLES:
                data=sorted((r for r in records if r["family"]==family and r["method"]==method),
                            key=lambda r:int(r["v"]))
                xs=[int(r["v"]) for r in data]
                for ax,ys in [(axes[0,column],[int(r["ops"]) for r in data]),
                              (axes[1,column],[1000*float(r["median_s"]) for r in data])]:
                    ax.loglog(xs,ys,color=color,marker=marker,linestyle=style,lw=1.4,ms=4,label=name)
                    ax.grid(axis="y",color="#d7d9db",lw=.5)
                    ax.spines[["top","right"]].set_visible(False)
                    ax.xaxis.set_minor_formatter(NullFormatter())
                    ax.tick_params(which="minor",bottom=False)
            axes[0,column].set_title(title,fontsize=10.5)
            axes[1,column].set_xticks(xs)
            axes[1,column].xaxis.set_major_formatter(ScalarFormatter())
            axes[1,column].set_xlabel("Kernel vertices")
        axes[0,0].set_ylabel("Arithmetic operations")
        axes[1,0].set_ylabel("Median time (ms)")
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=4,loc="lower center",frameon=False,fontsize=8,
                   handlelength=1.8,columnspacing=1.1,bbox_to_anchor=(0.5,0.006))
        fig.subplots_adjust(left=.12,right=.985,top=.93,bottom=.175,hspace=.16,wspace=.13)
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output)
        plt.close(fig)


def compact_accuracy_table(target):
    with (ROOT / "results/accuracy.csv").open() as source:
        rows = list(csv.DictReader(source))
    methods = {"positive":"Compressed positive", "full_positive":"Full positive",
               "compressed_cholesky":"Compressed Cholesky","full_cholesky":"Full Cholesky",
               "sherman_morrison":"Dense endpoint update","sparse_update":"Sparse endpoint update"}
    import math
    table = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & $s=0$ & $s=8$ & $s=16$ & $s=32$\\", r"\midrule"]
    for key, label in methods.items():
        cells = [label]
        for spread in [0, 8, 16, 32]:
            values = [float(row[key]) for row in rows if int(row["spread"]) == spread]
            finite = [value for value in values if math.isfinite(value)]
            mantissa, exponent = f"{max(finite):.2e}".split("e")
            cells.append(rf"${mantissa}\times10^{{{int(exponent)}}}$ ({len(values)-len(finite)})")
        table.append(" & ".join(cells) + r"\\")
    target.write_text("\n".join(table + [r"\bottomrule", r"\end{tabular}"]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT/"results/vertex_compiled.pdf")
    parser.add_argument("--output",dest="explicit_output",type=Path)
    options=parser.parse_args()
    output=options.explicit_output or options.output
    render(output)
    compact_accuracy_table(output.with_name("accuracy_table.tex"))
