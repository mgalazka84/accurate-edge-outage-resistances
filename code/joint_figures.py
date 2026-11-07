"""Generate the shared-reduction diagram, scaling figure, and table."""
import csv, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})


def scheme():
    with plt.rc_context({'mathtext.fontset':'stix', 'font.family':'DejaVu Sans'}):
        fig, ax = plt.subplots(figsize=(8.2, 3.5))
        ax.set(xlim=(0, 10), ylim=(0, 4.35))
        ax.axis('off')

        def box(x, y, w, h, label):
            patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                   facecolor='#f3f7fa', edgecolor='#185b70', linewidth=1.2)
            ax.add_patch(patch)
            ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                    linespacing=1.35, fontsize=11)

        box(2.9, 3.38, 4.2, .75, r'Queries $Q=Q_1\,\dot\cup\,Q_2$'+'\n'+
            r'Background $\mathcal{S}_Q$ with all edges of $Q$ absent')
        for side, x in enumerate((.15, 5.15), start=1):
            other = 3-side
            child = rf'Add $Q_{other}$ and retain $U_{{Q_{side}}}$'
            formula = rf'$\mathcal{{S}}_{{Q_{side}}}=\mathrm{{Sc}}_{{U_{{Q_{side}}}}}(\mathcal{{S}}_Q+L(Q_{other}))$'
            box(x, 1.88, 4.65, .83, child+'\n'+formula)
            h = 'h' if side == 1 else 'h\N{PRIME}'
            box(x+.3, .13, 4.05, .78,
                rf'A leaf for ${h}\in Q_{side}$'+'\n'+rf'$\beta_{{{h}}}=1/g_{{{h}}}$')
            center=x+2.325
            ax.annotate('',xy=(center,2.78),xytext=(4.45 if side==1 else 5.55,3.31),
                        arrowprops=dict(arrowstyle='->',color='#185b70',lw=1.3))
            ax.annotate('',xy=(center,.98),xytext=(center,1.8),
                        arrowprops=dict(arrowstyle='->',color='#185b70',lw=1.3))
        ax.text(5, 1.35, 'Repeat within each child', ha='center', fontsize=9)
        fig.tight_layout(pad=.25)
        fig.savefig(OUT/'shared_scheme.pdf')
        plt.close(fig)


def results():
    rows=list(csv.DictReader((ROOT/'results/joint_timings.csv').open()))
    matched=list(csv.DictReader((ROOT/'results/hra_comparison_timings.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.7))
    for method,label,style,color in [('independent','Independent positive','s--','#9b59b6'),('shared','Shared positive','o-','#087f8c')]:
        rr=[r for r in rows if r['family']=='ladder' and r['method']==method]
        xs=[int(r['rank']) for r in rr]
        axes[0].loglog(xs,[int(r['kernel_arithmetic_ops']) for r in rr],style,label=label,color=color)
        name={'independent':'compressed_positive','shared':'shared_positive'}[method]
        mm=[r for r in matched if r['family']=='ladder' and r['method']==name]
        axes[1].loglog([int(r['rank']) for r in mm],[float(r['median_seconds']) for r in mm],style,label=label,color=color)
    mm=[r for r in matched if r['family']=='ladder' and r['method']=='sf_ldl_compressed']
    axes[1].loglog([int(r['rank']) for r in mm],[float(r['median_seconds']) for r in mm],'x-.',label='Compressed SF-LDL',color='#267d43')
    for ax in axes:
        ax.set_xlabel('Cycle rank of circular ladder');ax.grid(alpha=.2)
    axes[0].set_ylabel('Kernel arithmetic operations')
    axes[1].set_ylabel('Seconds for all endpoint outages')
    axes[0].legend(frameon=False,fontsize=9)
    axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'shared_scaling.pdf');plt.close(fig)
    lines=[r'\begin{tabular}{rrrrrrrr}',r'\toprule',
        r'$v$ & $r$ & Independent ops & Shared ops & Ratio & Independent (s) & SF-LDL (s) & Shared (s)\\',r'\midrule']
    for n in [8,16,32,64,128]:
        a=next(r for r in rows if r['family']=='ladder' and int(r['n'])==n and r['method']=='independent')
        b=next(r for r in rows if r['family']=='ladder' and int(r['n'])==n and r['method']=='shared')
        x,y=int(a['kernel_arithmetic_ops']),int(b['kernel_arithmetic_ops'])
        times={r['method']:float(r['median_seconds']) for r in matched if r['family']=='ladder' and int(r['n'])==n}
        lines.append(f"{n} & {a['rank']} & {x:,} & {y:,} & {x/y:.2f} & {times['compressed_positive']:.6f} & {times['sf_ldl_compressed']:.6f} & {times['shared_positive']:.6f}"+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (OUT/'shared_table.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':scheme();results()
