"""Rebuild Figure 8 with shared reductions and matched complete-graph timings."""
import csv,io,os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results';OUT=ROOT/'manuscript'
def load(name):return list(csv.DictReader((RES/name).open()))
def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    styles={
        'compressed_positive':('#126a70','-','o','Algorithm 1: independent positive'),
        'shared_positive':('#702c91','-','*','Algorithm 2: shared positive'),
        'compressed_cholesky':('#db9a30','-.','D','Compressed Cholesky'),
        'full_positive':('#69a7a0','--','v','Full positive'),
        'full_cholesky':('#92509e',':','P','Full Cholesky'),
        'sherman_morrison':('#b54a47','-','s','Dense factor + endpoint formula'),
        'sparse_update':('#3f62a0','--','^','Sparse LU + endpoint formula')}
    left=load('timings.csv')+load('positioning_subdivision_timings.csv')
    right=load('positioning_complete_timings.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.2))
    for ax,rows,key in [(axes[0],left,'m'),(axes[1],right,'rank')]:
        for name,(color,ls,mark,label) in styles.items():
            rr=[r for r in rows if r['method']==name]
            if rr:ax.loglog([int(r[key]) for r in rr],[float(r['median_seconds']) for r in rr],color=color,ls=ls,marker=mark,label=label,ms=6 if name=='shared_positive' else 4,lw=1.6 if name=='shared_positive' else 1.1)
        ax.grid(alpha=.18)
    axes[0].set(xlabel='Edges in subdivided $K_4$ (cycle rank 3)',ylabel='Seconds for all endpoint outages',title='Fixed rank: growing chains')
    axes[1].set(xlabel='Cycle rank of complete graph',ylabel='Seconds for all endpoint outages',title='No chain reduction')
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.01),ncol=2,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.24,1,1))
    buf=io.BytesIO();fig.savefig(buf,format='pdf',bbox_inches='tight')
    tmp=OUT/'runtime_scaling.pdf.tmp'
    with tmp.open('wb') as f:f.write(buf.getvalue());f.flush();os.fsync(f.fileno())
    tmp.replace(OUT/'runtime_scaling.pdf')
    fig.savefig(OUT/'runtime_scaling.png',dpi=180,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
