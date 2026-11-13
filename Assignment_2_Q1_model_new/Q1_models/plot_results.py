"""Plot three economic metrics for the discount and six metrics for each other sweep.
Run: python plot_results.py
"""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'results'/'sensitivity'
LABELS={'R':'R: Reference case','I':'I: Individual contracts model','C':'C: Community contract model'}
STYLES={'R':dict(color='#111111',linestyle='--',linewidth=1.8),'I':dict(color='#cc1526',linestyle='-',linewidth=1.8),'C':dict(color='#0057b8',linestyle='-.',linewidth=1.8,marker='o',markersize=3.6,markerfacecolor='white',markevery=10)}
XLABELS={'beta':r'Tariff discount $\beta^{\mathrm{DSO}}$','f':r'Cap factor $f$ (base cap $\times f/1.3$)','alpha':r'Penalty $\alpha^{\mathrm{DSO}}$ (DKK/kWh)'}
PANELS=[('J','(a) Total community cost','Total cost (DKK)'),
        ('gain_vs_R','(b) Savings relative to no contract','Savings (DKK)'),
        ('pooling_gain','(c) Collective vs individual savings','Additional savings (DKK)'),
        ('V','(d) Total cap exceedance','Exceedance (kWh)'),
        ('peak','(e) Peak import','Peak import (kW)'),
        ('H','(f) Hours above the cap','Hours (h)')]

def plot():
    rows=list(csv.DictReader((ROOT/'sensitivity_metrics.csv').open(encoding='utf-8')))
    thresholds=json.loads((ROOT/'thresholds.json').read_text(encoding='utf-8'))
    for p in ['beta','f','alpha']:
        panels = PANELS[:3] if p == 'beta' else PANELS
        nrows = len(panels) // 3
        for full in ([False] if p=='beta' else [False,True]):
            lim={'beta':(0,1),'f':(.65,8 if full else 3),'alpha':(0,30 if full else 6)}[p]
            with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,
                                 'axes.titlesize':11.5,'pdf.fonttype':42}):
                fig,axs=plt.subplots(nrows,3,figsize=(14.6,4.4 if nrows == 1 else 8.2),
                                     sharex=True,squeeze=False)
                for m in ['R','I','C']:
                    rr=sorted([r for r in rows if r['parameter']==p and r['model']==m and lim[0]<=float(r['value'])<=lim[1]],key=lambda r:float(r['value']))
                    xx=[float(r['value']) for r in rr]
                    for ax,(field,title,ylabel) in zip(axs.flat,panels):
                        if field=='pooling_gain' and m!='C':continue
                        yy=[float(r[field]) for r in rr]
                        if field=='H':ax.step(xx,yy,where='post',label=LABELS[m],**STYLES[m])
                        else:ax.plot(xx,yy,label=LABELS[m],**STYLES[m])
                for ax,(field,title,ylabel) in zip(axs.flat,panels):
                    ax.set(title=title,ylabel=ylabel,xlim=lim)
                    ax.spines[['top','right']].set_visible(False)
                    ax.grid(axis='y',alpha=.25)
                    if field in ['V','H']:
                        ax.set_ylim(bottom=-.4 if field=='H' else -max(1,ax.get_ylim()[1]*.025))
                    if field=='H':ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                for ax in axs[-1,:]:ax.set_xlabel(XLABELS[p])
                for t in thresholds:
                    if t['parameter']!=p or not lim[0]<=t['value']<=lim[1]:continue
                    # Signing lines go in cost and savings panels; penalty plateaus in DSO panels.
                    targets=[axs[0,0],axs[0,1]] if t['kind']=='signing' else list(axs[1,:])
                    for ax in targets:ax.axvline(t['value'],color='#999999',linestyle=':',linewidth=.8)
                    if t['kind']=='signing':
                        axs[0,0].annotate(f"{t['model']}: {t['value']:.4f}",
                            (t['value'],t['target']),
                            xytext=(7,14 if t['model']=='C' else -26),textcoords='offset points',
                            fontsize=9,arrowprops=dict(arrowstyle='-',color='#777777',linewidth=.7),
                            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
                h,l=axs[0,0].get_legend_handles_labels()
                fig.legend(h,l,loc='lower center',ncol=3,frameon=False,
                           bbox_to_anchor=(.5,.012),fontsize=10,handlelength=3.2)
                fig.tight_layout(rect=[0,.13 if nrows == 1 else .075,1,1],w_pad=2.2,h_pad=2)
                name=f'sensitivity_{p}'+('_full' if full else '')
                for ext in ['png','pdf']:fig.savefig(ROOT/(name+'.'+ext),dpi=300,bbox_inches='tight')
                plt.close(fig)

if __name__=='__main__':plot()
