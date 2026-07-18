from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(__file__).resolve().parents[1]/'results'/'sncs'/'merged_results.json'
OUT=ROOT/'02_Figures'; OUT.mkdir(exist_ok=True)
d=json.loads(DATA.read_text())['summary']

plt.rcParams.update({
    'font.size':9,
    'axes.labelsize':9,
    'font.family':'DejaVu Sans',
    'figure.dpi':160,
    'savefig.dpi':600,
    'pdf.fonttype':42,
    'ps.fonttype':42,
})

def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)

# Fig1 architecture
fig,ax=plt.subplots(figsize=(10,3.7)); ax.axis('off')
boxes=[('Problem\n& grammar',0.03),('Parent\nprograms',0.20),('PoW share or\nPRNG seed',0.37),('Domain-separated\nseed derivation',0.55),('Bounded mutation\n& crossover',0.73),('Evaluation, cache\n& selection',0.89)]
for label,x in boxes:
    ax.text(x,0.62,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=0.42',fc='white',ec='black',lw=1.2),transform=ax.transAxes)
for (_,x1),(_,x2) in zip(boxes,boxes[1:]):
    ax.annotate('',xy=(x2-0.07,0.62),xytext=(x1+0.07,0.62),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',lw=1.3))
ax.annotate('next generation',xy=(0.20,0.43),xytext=(0.89,0.24),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',lw=1.2,connectionstyle='arc3,rad=-0.25'))
ax.text(0.63,0.91,'proof-of-work qualification',ha='center',va='center',transform=ax.transAxes,weight='bold')
ax.text(0.88,0.91,'semantic evaluation outside ASIC',ha='center',va='center',transform=ax.transAxes,weight='bold')
save(fig,'Fig1_Architecture')

# Fig2 primary paired outcomes (no internal title; caption is in manuscript)
c=d['primary_300_paired']['vhss_hash_vs_prng']
vals=[c['a_wins'],c['ties'],c['a_losses']]; labs=['SHA-256 wins','Ties','PRNG wins']
fig,ax=plt.subplots(figsize=(7.2,4.0)); bars=ax.bar(labs,vals,edgecolor='black')
for i,b in enumerate(bars):
    b.set_hatch(['///','..','\\\\'][i])
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+3,str(int(b.get_height())),ha='center')
ax.set_ylabel('Paired runs')
ax.set_ylim(0,max(vals)*1.18); ax.grid(axis='y',alpha=.25)
ax.text(0.02,0.97,'Sign test p = 0.110; Wilcoxon test-error p = 0.245',transform=ax.transAxes,va='top')
save(fig,'Fig2_Hash_vs_PRNG')

# Fig3 exact rates with Wilson CIs
methods=['vhss_hash','vhss_prng','standard_gp','beam_mutation','simulated_annealing','random_program']
labels=['VHSS-SHA256','VHSS-PRNG','Standard GP','Beam mutation','Simulated annealing','Random program']
ms=d['baseline_90_paired']['methods']
rates=[ms[m]['test_exact_rate']*100 for m in methods]
lo=[(ms[m]['test_exact_rate']-ms[m]['test_exact_wilson95'][0])*100 for m in methods]
hi=[(ms[m]['test_exact_wilson95'][1]-ms[m]['test_exact_rate'])*100 for m in methods]
fig,ax=plt.subplots(figsize=(8.8,4.5)); y=np.arange(len(methods))
bars=ax.barh(y,rates,xerr=np.array([lo,hi]),capsize=4,edgecolor='black')
for i,b in enumerate(bars): b.set_hatch(['///','\\\\','xx','..','++','oo'][i])
ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(0,105); ax.set_xlabel('Exact recovery on held-out test inputs (%)')
for yy,v in zip(y,rates): ax.text(min(v+1,98),yy,f'{v:.1f}%',va='center')
ax.grid(axis='x',alpha=.25)
save(fig,'Fig3_Exact_Recovery')

# Fig4 median proposals and runtime
fig,ax=plt.subplots(figsize=(8.2,4.5))
x=[ms[m]['median_proposals'] for m in methods]; y=[ms[m]['median_elapsed_seconds'] for m in methods]
for i,(xx,yy,l) in enumerate(zip(x,y,labels)):
    ax.scatter(xx,yy,s=70,marker=['o','s','^','D','P','X'][i],edgecolor='black')
    ax.annotate(l,(xx,yy),xytext=(5,5),textcoords='offset points',fontsize=8)
ax.set_xlabel('Median proposals per run'); ax.set_ylabel('Median measured software time (s)'); ax.grid(alpha=.25)
save(fig,'Fig4_Resource_Profile')

# Fig5 seed derivation
fig,ax=plt.subplots(figsize=(9,3.8)); ax.axis('off')
ax.text(.18,.65,'Accepted share digest d\n(target condition may bias prefix bits)',ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',fc='white',ec='black'),transform=ax.transAxes)
ax.text(.50,.65,'SHA-256(domain || d || nonce || parent || task || generation)',ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',fc='white',ec='black'),transform=ax.transAxes)
ax.text(.83,.65,'Unconstrained 256-bit\nproposal seed u',ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',fc='white',ec='black'),transform=ax.transAxes)
ax.annotate('',xy=(.36,.65),xytext=(.30,.65),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->'))
ax.annotate('',xy=(.71,.65),xytext=(.65,.65),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->'))
ax.text(.50,.25,'Context binding separates work qualification from proposal selection',ha='center',transform=ax.transAxes)
save(fig,'Fig5_Seed_Derivation')

# Fig6 hardware validation protocol
fig,ax=plt.subplots(figsize=(10,4.0)); ax.axis('off')
steps=['Isolated\nStratum server','SHA-256\nASIC miner','Share capture\n& replay checks','Seed and proposal\nderivation','CPU semantic\nevaluation','Energy and\nthroughput report']
xs=np.linspace(.07,.93,len(steps))
for i,(x,label) in enumerate(zip(xs,steps)):
    ax.text(x,.62,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.4',fc='white',ec='black',lw=1.1),transform=ax.transAxes)
    if i<len(steps)-1: ax.annotate('',xy=(xs[i+1]-.07,.62),xytext=(x+.07,.62),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->'))
ax.text(.5,.22,'accepted shares/s  |  stale and duplicate rates  |  latency  |  candidates/kWh  |  exact solutions/kWh  |  CPU bottlenecks',ha='center',transform=ax.transAxes)
save(fig,'Fig6_Hardware_Validation')

print(OUT)
