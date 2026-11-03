"""Build manuscript figures and tables directly from recorded CSV results."""
import csv,json,math,statistics,io,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'; OUT=ROOT/'manuscript'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                    'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
COL={'compressed_positive':'#126a70','full_positive':'#69a7a0',
     'compressed_cholesky':'#db9a30','full_cholesky':'#92509e',
     'sherman_morrison':'#b54a47','sparse_update':'#3f62a0'}
LABEL={'compressed_positive':'Compressed positive','full_positive':'Full positive',
       'compressed_cholesky':'Compressed Cholesky','full_cholesky':'Full Cholesky',
       'sherman_morrison':'Dense factor + endpoint formula','sparse_update':'Sparse LU + endpoint formula'}
STYLE={'compressed_positive':('-', 'o'),'full_positive':('--','v'),
       'compressed_cholesky':('-.','D'),'full_cholesky':(':','P'),
       'sherman_morrison':('-','s'),'sparse_update':('--','^')}

def load(name): return list(csv.DictReader((RES/name).open()))
def save(fig,name):
    # Finish the PDF in memory before exposing it to a TeX or preview reader.
    buffer=io.BytesIO();fig.savefig(buffer,format='pdf')
    temporary=OUT/(name+'.pdf.tmp')
    with temporary.open('wb') as handle:
        handle.write(buffer.getvalue());handle.flush();os.fsync(handle.fileno())
    temporary.replace(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)
def sci(x):
    if x==0:return '0'
    a,b=f'{x:.2e}'.split('e');return rf'${a}\times10^{{{int(b)}}}$'

def main():
    OUT.mkdir(exist_ok=True)
    a=load('accuracy.csv');tim=load('timings.csv');rank=load('rank_timings.csv')
    summary={}
    table=['\\begin{tabular}{llrrr}','\\toprule','Spread & Method & Queries & Failed & Max. finite rel. error\\\\','\\midrule']
    for spread in [0,8,16,32]:
        rr=[r for r in a if int(r['spread'])==spread];summary[str(spread)]={}
        for key in ['positive','full_positive','compressed_cholesky','full_cholesky','sherman_morrison','sparse_update']:
            vals=[float(r[key]) for r in rr];finite=[v for v in vals if math.isfinite(v)]
            s=dict(queries=len(vals),failed=len(vals)-len(finite),above_1e8=sum(v>1e-8 for v in vals),
                   max_finite=max(finite,default=math.nan),median_finite=statistics.median(finite) if finite else math.nan)
            summary[str(spread)][key]=s
            label={'positive':'Compressed positive','full_positive':'Full positive','compressed_cholesky':'Compressed Cholesky',
                   'full_cholesky':'Full Cholesky','sherman_morrison':'Dense endpoint update','sparse_update':'Sparse endpoint update'}[key]
            table.append(f'{spread if key=="positive" else ""} & {label} & {len(vals)} & {s["failed"]} & {sci(s["max_finite"])}\\\\')
        if spread!=32:table.append('\\addlinespace')
    table.extend(['\\bottomrule','\\end{tabular}'])
    (OUT/'accuracy_table.tex').write_text('\n'.join(table)+'\n')
    summary['overall']=dict(graphs=100,queries=len(a),max_positive=max(float(r['positive']) for r in a),
                            max_defect=max(float(r['positive_defect']) for r in a),
                            max_bound_ratio=max(float(r['positive'])/float(r['analytic_bound']) for r in a),
                            min_n=min(int(r['n']) for r in a),max_n=max(int(r['n']) for r in a))
    (RES/'summary.json').write_text(json.dumps(summary,indent=2))

    fig,ax=plt.subplots(figsize=(6.9,3.2))
    tri=load('triangle.csv');x=[int(r['exponent']) for r in tri]
    for key,label,col,mark in [('positive_defect_relerr','Positive quotient','#126a70','o'),
                              ('oracle_subtraction_relerr','Subtract rounded exact resistance','#b54a47','s')]:
        y=[max(float(r[key]),1e-18) if math.isfinite(float(r[key])) else 1 for r in tri]
        ax.semilogy(x,y,marker=mark,color=col,label=label)
    ax.axhline(2**-53,color='#999999',ls=':',lw=1,label='Unit roundoff')
    ax.set(xlabel=r'$k$ in resistances $(1,10^k,10^k)$',ylabel=r'Relative error in $\delta_e$',ylim=(1e-18,3))
    ax.grid(alpha=.18);ax.legend(loc='center right',fontsize=8)
    save(fig,'triangle_accuracy')

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1))
    methods=['positive','full_positive','compressed_cholesky','full_cholesky','sherman_morrison','sparse_update']
    for key in methods:
        colour=COL.get(key,COL['compressed_positive']);label=LABEL.get(key,'Compressed positive')
        ls,marker=STYLE['compressed_positive' if key=='positive' else key]
        med=[];bad=[]
        for spread in [0,8,16,32]:
            rr=[float(r[key]) for r in a if int(r['spread'])==spread]
            # Failed results receive unit error for this display only; raw tables retain failure flags.
            med.append(statistics.median([min(v,1.0) for v in rr]))
            bad.append(100*sum(v>1e-8 for v in rr)/len(rr))
        axes[0].semilogy([0,8,16,32],np.maximum(med,1e-18),linestyle=ls,marker=marker,color=colour,label=label,markersize=4)
        axes[1].plot([0,8,16,32],bad,linestyle=ls,marker=marker,color=colour,markersize=4)
    bounds=[[float(r['analytic_bound']) for r in a if int(r['spread'])==s] for s in [0,8,16,32]]
    axes[0].semilogy([0,8,16,32],[statistics.median(b) for b in bounds],color='#222222',ls=':',lw=1.7,label='Median proven bound')
    axes[0].fill_between([0,8,16,32],[min(b) for b in bounds],[max(b) for b in bounds],color='#999999',alpha=.13)
    axes[0].set(ylabel='Median relative error',xlabel='Log-uniform half-width $s$')
    axes[1].set(ylabel='Errors > $10^{-8}$ or failures (%)',xlabel='Log-uniform half-width $s$',ylim=(-3,103))
    for ax in axes:ax.grid(alpha=.18);ax.set_xticks([0,8,16,32])
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8)
    fig.tight_layout();save(fig,'weight_accuracy')

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2))
    for method in COL:
        rows=[r for r in tim if r['method']==method]
        ls,marker=STYLE[method]
        axes[0].loglog([int(r['m']) for r in rows],[float(r['median_seconds']) for r in rows],
                      linestyle=ls,marker=marker,color=COL[method],label=LABEL[method],markersize=4)
    for method in ['compressed_positive','sherman_morrison','sparse_update']:
        rows=[r for r in rank if r['method']==method]
        ls,marker=STYLE[method]
        axes[1].loglog([int(r['rank']) for r in rows],[float(r['median_seconds']) for r in rows],
                      linestyle=ls,marker=marker,color=COL[method],markersize=4)
    axes[0].set(xlabel='Edges in subdivided $K_4$ (cycle rank 3)',ylabel='Seconds for all endpoint outages')
    axes[1].set(xlabel='Cycle rank of complete graph',ylabel='Seconds for all endpoint outages')
    for ax in axes:ax.grid(alpha=.18)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8)
    fig.tight_layout();save(fig,'runtime_scaling')

    table=['\\begin{tabular}{rrrrrr}','\\toprule',
           '$m$ & $n$ & Compressed positive & Compressed Cholesky & Full positive & Sparse update\\\\','\\midrule']
    for length in [1,4,16,64,256,1024,4096,16384]:
        rr=[r for r in tim if int(r['length'])==length];d={r['method']:float(r['median_seconds']) for r in rr}
        vals=[f'{int(rr[0]["m"]):,}',f'{int(rr[0]["n"]):,}']+[f'{d[k]:.6f}' if k in d else '--' for k in
            ['compressed_positive','compressed_cholesky','full_positive','sparse_update']]
        table.append(' & '.join(vals)+'\\\\')
    table.extend(['\\bottomrule','\\end{tabular}'])
    (OUT/'timing_table.tex').write_text('\n'.join(table)+'\n')

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2))
    ax=axes[0]
    for y in [-1,0,1]:
        pts=[(0,0),(1,y),(2,y),(3,0)]
        for idx,((x1,y1),(x2,y2)) in enumerate(zip(pts,pts[1:])):
            ax.plot([x1,x2],[y1,y2],color='#b54a47' if y==1 and idx==1 else '#126a70',lw=2,
                    ls='--' if y==1 and idx==1 else '-')
        for x1,y1 in pts[1:3]:ax.add_patch(Circle((x1,y1),.06,color='#126a70'))
    for x in [0,3]:ax.add_patch(Circle((x,0),.09,color='#126a70'))
    for x,y,lab in [(0.45,.77,'$e_1: 1$'),(1.5,1.18,'$e_2: 2$'),(2.55,.77,'$e_3: 3$')]:
        ax.text(x,y,lab,ha='center',fontsize=9,color='#b54a47' if x==1.5 else '#222222')
    ax.text(1.5,.16,'Middle chain: $S_2=4$',ha='center',fontsize=8)
    ax.text(1.5,-1.35,'Lower chain: $S_3=12$',ha='center',fontsize=8)
    ax.text(1.5,-1.9,'Example shown: remove $e_2$; excluded sum = 4',ha='center',fontsize=8)
    ax.set_title('(a) Three separate outages in one chain',fontsize=9)
    ax=axes[1]
    for y,lab in [(-.7,'$S_3=12$'),(.7,'$S_2=4$')]:
        ax.plot([0,1.5,3],[0,y,0],color='#126a70',lw=2)
        ax.text(1.5,y+(.15 if y>0 else -.3),lab,ha='center')
    for x in [0,3]:ax.add_patch(Circle((x,0),.09,color='#126a70'))
    ax.text(1.5,-1.35,r'$\beta=(1/4+1/12)^{-1}=3$',ha='center',fontsize=10)
    ax.text(1.5,-1.9,r'$\rho_{e_1}=8\qquad\rho_{e_2}=7\qquad\rho_{e_3}=6$',ha='center',fontsize=10)
    ax.set_title('(b) One bypass value supplies all three',fontsize=9)
    for ax in axes:ax.set(xlim=(-.3,3.3),ylim=(-2.2,1.6));ax.axis('off')
    fig.tight_layout();save(fig,'series_identity')

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.05))
    for i,ax in enumerate(axes):
        left=(.3,.15);right=(2.7,.15);top=(1.5,1.55)
        ax.plot([left[0],top[0],right[0]],[left[1],top[1],right[1]],color='#126a70',lw=2.2)
        ax.plot([left[0],right[0]],[left[1],right[1]],color='#222222' if i==0 else '#b54a47',
                lw=2.2 if i==0 else 1.7,ls='-' if i==0 else '--')
        for x,y in [left,right,top]:ax.add_patch(Circle((x,y),.055,color='#222222'))
        for x,lab in [(.3,'$u$'),(2.7,'$v$')]:ax.text(x,-.12,lab,ha='center',fontsize=10)
        ax.text(.73,1.01,'$M$',ha='center',fontsize=11)
        ax.text(2.27,1.01,'$M$',ha='center',fontsize=11)
        if i==0:
            ax.text(1.5,-.12,'$e$: resistance 1',ha='center',fontsize=9)
            ax.text(1.5,-.66,r'$R_e=2M/(1+2M)$',ha='center',fontsize=11)
        else:
            ax.plot(1.5,.15,marker='x',ms=8,color='#b54a47',mew=1.8)
            ax.text(1.5,-.12,'deleted $e$',ha='center',fontsize=9,color='#b54a47')
            ax.text(1.5,-.66,r'$\rho_e=2M\quad\mathrm{(finite)}$',ha='center',fontsize=11)
        ax.set(xlim=(-.05,3.05),ylim=(-.92,1.85));ax.set_aspect('equal');ax.axis('off')
    axes[0].set_title('(a) Direct connection and weak bypass',fontsize=9)
    axes[1].set_title('(b) After failure: still connected',fontsize=9)
    fig.text(.5,-.015,r'At $M=10^{16}$: $1-\mathrm{fl}(R_e)=0$, while $1/(1+2M)\approx5\times10^{-17}$.',
             ha='center',fontsize=10)
    fig.tight_layout();save(fig,'outage_motivation')


if __name__=='__main__':main()
