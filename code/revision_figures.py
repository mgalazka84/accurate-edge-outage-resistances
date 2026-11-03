"""Figures, tables and numerical summaries for the major revision."""
import csv,io,json,math,os,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Arc
from revision_experiments import atomic

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; OUT=ROOT/'manuscript'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                    'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})

def load(name): return list(csv.DictReader((RES/name).open()))
def sci(x):
    if x==0: return '$0$'
    a,b=f'{x:.2e}'.split('e'); return rf'${a}\times10^{{{int(b)}}}$'
def save(fig,name):
    for ext in ['pdf','png']:
        b=io.BytesIO(); fig.savefig(b,format=ext,dpi=170); atomic(OUT/(name+'.'+ext),b.getvalue())
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    rows=load('revision_accuracy.csv')
    labels={'sf_ldl_full':'SF-LDL, full graph','sf_ldl_compressed':'SF-LDL, compressed',
        'decimal35_direct':'Decimal 35, direct','decimal80_direct':'Decimal 80, direct',
        'decimal35_update':'Decimal 35, update','cholesky_ir':'Cholesky + refinement'}
    summary={'accuracy':{},'rank':{},'pglib':{},'timing_slopes':{}}
    table=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & $s=0$ & $s=8$ & $s=16$ & $s=32$\\',r'\midrule']
    for method,label in labels.items():
        cells=[]; summary['accuracy'][method]={}
        for spread in [0,8,16,32]:
            vals=[float(r[method]) for r in rows if int(r['spread'])==spread]
            finite=[x for x in vals if math.isfinite(x)]; failed=len(vals)-len(finite)
            stats=dict(queries=len(vals),failed=failed,max_finite=max(finite),median_finite=statistics.median(finite),above_1e8=sum(x>1e-8 for x in vals))
            summary['accuracy'][method][str(spread)]=stats
            cells.append(sci(max(finite))+f' ({failed})')
        table.append(label+' & '+' & '.join(cells)+r'\\')
    table.extend([r'\bottomrule',r'\end{tabular}'])
    atomic(OUT/'revision_accuracy_table.tex','\n'.join(table)+'\n')

    rank=load('revision_rank.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.2))
    ranks=[]; exponents=[]; bounds=[]
    for n in [4,8,16,32,64]:
        rr=[r for r in rank if int(r['n'])==n]
        ranks.append(int(rr[0]['rank'])); exponents.append(int(rr[0]['exponent']));bounds.append(float(rr[0]['bound']))
        summary['rank'][str(n)]=dict(rank=ranks[-1],exponent=exponents[-1],bound=bounds[-1],
            max_error=max(float(r['relative_error']) for r in rr),max_error_to_bound=max(float(r['error_to_bound']) for r in rr))
    axes[0].plot(ranks,exponents,'o-',color='#126a70')
    axes[0].set(xlabel='Component cycle rank $r_i$',ylabel='Proven exponent $E$')
    for x,y,n in zip(ranks,exponents,[4,8,16,32,64]):
        offset=(0,13) if n==4 else (5,-10) if n==8 else (4,5)
        axes[0].annotate(f'$v={n}$',(x,y),xytext=offset,textcoords='offset points',fontsize=8)
    axes[0].set_ylim(0,650);axes[0].set_xlim(0,37)
    axes[1].semilogy(ranks,bounds,'k:',label=r'Proven $\nu^E-1$ (approximately $Eu$)')
    for s,c,m in [(0,'#126a70','o'),(8,'#b54a47','s')]:
        vals=[max(float(r['relative_error']) for r in rank if int(r['rank'])==rr and int(r['spread'])==s) for rr in ranks]
        axes[1].semilogy(ranks,[x if x>=1e-17 else np.nan for x in vals],marker=m,color=c,label=f'Maximum error, $s={s}$')
        for rr,value in zip(ranks,vals):
            if value<1e-17:
                axes[1].annotate('$<10^{-17}$',(rr,1.5e-17),fontsize=7,color=c,ha='center')
    axes[1].set(xlabel='Component cycle rank $r_i$',ylabel='Relative error or bound',ylim=(1e-17,2e-13)); axes[1].legend(fontsize=7.5)
    for ax in axes: ax.grid(alpha=.2);ax.set_xticks(ranks)
    fig.tight_layout();save(fig,'bound_rank')

    grids=json.loads((RES/'revision_pglib_summary.json').read_text()); gr=load('revision_pglib_accuracy.csv')
    tab=[r'\begin{tabular}{rrrrrrr}',r'\toprule',r'Buses & Edges & Bridges & Kernel $(v,k)$ & $r_i$ & Min. $\delta_e$ & Max. rel. error\\',r'\midrule']
    for g in grids:
        rr=[r for r in gr if r['case']==g['case']]
        stats=dict(finite_queries=len(rr),max_positive=max(float(r['compressed_positive']) for r in rr),min_delta=min(float(r['delta_reference']) for r in rr),max_defect=max(float(r['positive_defect']) for r in rr))
        summary['pglib'][g['case']]=stats
        tab.append(f"{g['n']} & {g['m']} & {g['bridges']} & $({g['kernel_vertices'][0]},{g['kernel_edges'][0]})$ & {g['kernel_ranks'][0]} & {stats['min_delta']:.3f} & {sci(stats['max_positive'])}"+r'\\')
    tab.extend([r'\bottomrule',r'\end{tabular}']);atomic(OUT/'pglib_table.tex','\n'.join(tab)+'\n')

    timings=load('revision_timings.csv');gt=load('revision_pglib_timings.csv')
    names={'compressed_positive':'Compressed positive',**labels,'sparse_update':'Sparse endpoint update'}
    colors={'compressed_positive':'#126a70','sf_ldl_compressed':'#69a7a0','sf_ldl_full':'#92509e',
            'decimal35_direct':'#bd9c38','decimal80_direct':'#cb5a35','decimal35_update':'#3f62a0',
            'cholesky_ir':'#666666','sparse_update':'#b54a47'}
    marks=['o','D','s','v','^','P','x','+']; styles=['-','--','-.',':','--','-.',':','-']
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.3));handles=[]
    for j,(name,color) in enumerate(colors.items()):
        data=[r for r in timings if r['method']==name]
        if data:
            line,=axes[0].loglog([int(r['m']) for r in data],[float(r['median_seconds']) for r in data],linestyle=styles[j],marker=marks[j],color=color,ms=4,label=names[name]);handles.append(line)
        data=[r for r in gt if r['method']==name]
        if data:
            line,=axes[1].loglog([int(r['case'].split('case')[1].split('_')[0]) for r in data],[float(r['median_seconds']) for r in data],linestyle=styles[j],marker=marks[j],color=color,ms=4,label=names[name])
            if name=='sparse_update': handles.append(line)
    axes[0].set(xlabel='Edges in subdivided $K_4$',ylabel='Seconds for all endpoint outages')
    axes[1].set(xlabel='Buses in PGLib IEEE case',ylabel='Seconds for all endpoint outages')
    for ax in axes: ax.grid(alpha=.2)
    fig.legend(handles,[h.get_label() for h in handles],loc='lower center',bbox_to_anchor=(.5,-.27),ncol=2,fontsize=8)
    fig.tight_layout();save(fig,'revision_runtime')
    # Fit only the large-subdivision region of the original scaling experiment.
    for name in ['compressed_positive','compressed_cholesky']:
        rr=[r for r in load('timings.csv') if r['method']==name and int(r['length'])>=256]
        slope=float(np.polyfit(np.log([int(r['m']) for r in rr]),np.log([float(r['median_seconds']) for r in rr]),1)[0])
        summary['timing_slopes'][name]=dict(log_log_slope=slope,min_m=min(int(r['m']) for r in rr),max_m=max(int(r['m']) for r in rr),points=len(rr))

    fig,axes=plt.subplots(1,2,figsize=(7.4,2.7))
    a=axes[0];points=[(.2,.1),(1.5,1.5),(2.8,.1)]
    for j,(v,w) in enumerate([(0,1),(1,2),(2,0)]):
        p,q=points[v],points[w]; a.plot([p[0],q[0]],[p[1],q[1]],color='#b54a47' if j==2 else '#126a70',ls='--' if j==2 else '-',lw=2)
    for p in points: a.add_patch(Circle(p,.055,color='#222222'))
    for x,y,txt in [(.6,1,'2'),(2.4,1,'3'),(1.5,-.1,'deleted edge: 1')]:a.text(x,y,txt,ha='center')
    a.text(1.5,-.58,r'$\rho=2+3=5,\quad\beta=0$',ha='center',fontsize=11)
    a.set_title('(a) A rank-one cycle',fontsize=10)
    a=axes[1];a.add_patch(Arc((1.1,.65),1.5,1.5,theta1=0,theta2=360,color='#126a70',lw=2))
    a.add_patch(Circle((1.85,.65),.07,color='#222222'));a.text(2.02,.7,'anchor',fontsize=8)
    a.plot([1.85,2.7,2.7,1.85],[.65,1.4,-.1,.65],color='#999999',lw=1.6)
    a.text(.9,.65,r'$h$',fontsize=13);a.text(2.65,-.4,'other network',ha='center',fontsize=8,color='#666666')
    a.text(1.65,-.9,'One anchor: attachments carry zero current\nfor an outage inside the closed chain.',ha='center',fontsize=9)
    a.set_title('(b) A compressed loop, possibly attached',fontsize=10)
    for a in axes:a.set(xlim=(-.1,3.4),ylim=(-1.15,1.8));a.axis('off');a.set_aspect('equal')
    fig.tight_layout();save(fig,'cycle_case')
    atomic(RES/'revision_summary.json',json.dumps(summary,indent=2)+'\n')

if __name__=='__main__': main()
