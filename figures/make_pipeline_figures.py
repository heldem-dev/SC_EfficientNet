# Regenerate the overall experimental pipeline figure.
# Dependencies: matplotlib
# The script intentionally uses monochrome publication-style boxes/lines.
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "figures"
OUT.mkdir(exist_ok=True)

def box(ax,x,y,w,h,text,fs=9.5):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.02',linewidth=1.4,facecolor='white',edgecolor='black')
    ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,wrap=True,linespacing=1.15)

def arr(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,linewidth=1.15,color='black'))

# Figure 1
fig,ax=plt.subplots(figsize=(14,9)); ax.set_xlim(0,14); ax.set_ylim(0,10); ax.axis('off')
ax.text(7,9.65,'Overall experimental and evaluation pipeline',ha='center',fontsize=18,fontweight='bold')
items=[(0.5,7.9,2,.85,'ISIC 2018 Task 3\n10,015 images • 7 classes'),(2.9,7.9,2.2,.85,'Metadata + lesion QC\n7,470 lesions\nno missing IDs'),(5.5,7.9,2.2,.85,'StratifiedGroupKFold\n5 folds • seed = 42\nlesion_id = group'),(8.1,7.9,2.2,.85,'Fold-specific evaluation\n4 folds train\n1 fold validation')]
for a in items: box(ax,*a)
for a,b in [((2.5,8.325),(2.9,8.325)),((5.1,8.325),(5.5,8.325)),((7.7,8.325),(8.1,8.325))]: arr(ax,a,b)
for a in [(0.7,6.1,2.6,1,'B0 ablations\n224×224\nCE / CE+KL / Focal+KL\nλ sensitivity'),(3.8,6.1,2.6,1,'Sampling ablation\nstandard baseline vs.\nWeightedRandomSampler'),(6.9,6.1,2.6,1,'Final SC-EfficientNet-B4\n384×384 • CE+KL\nλ = 0.10 • weighted sampler'),(10.0,6.1,2.7,1,'Controlled analyses\nTTA • artifacts • XAI')]: box(ax,*a)
# branch arrows from fold node
for end in [(2.0,7.1),(5.1,7.1),(8.2,7.1),(11.35,7.1)]: arr(ax,(9.2,7.9),end)
for a in [(0.7,4.35,2.8,1,'Training preprocessing\nResize → H/V flip → ColorJitter\nTensor → ImageNet normalization'),(3.95,4.35,2.6,1,'SC-EfficientNet\nEfficientNet-B0/B4 backbone\nspatial attention mask M'),(7.0,4.35,2.65,1,'Dual prediction paths\nP_org and P_masked\njoint training'),(10.1,4.35,2.65,1,'Training objective\nL = L_cls + 0.10 L_cons\nKL(P_masked || stopgrad(P_org))')]: box(ax,*a)
for a,b in [((3.5,4.85),(3.95,4.85)),((6.55,4.85),(7,4.85)),((9.65,4.85),(10.1,4.85))]: arr(ax,a,b)
for a in [(1.3,2.55,2.6,.95,'Inference\nuse P_masked'),(4.4,2.55,2.8,.95,'Five-view TTA\noriginal + H + V + HV\n+10° rotation'),(7.8,2.55,2.6,.95,'Mean softmax probability\n→ final class'),(10.9,2.55,2,.95,'OOF predictions')]: box(ax,*a)
for a,b in [((2.5,4.35),(2.5,3.5)),((3.9,3.025),(4.4,3.025)),((7.2,3.025),(7.8,3.025)),((10.4,3.025),(10.9,3.025))]: arr(ax,a,b)
outs=[(.5,'Classification metrics\nAccuracy • Macro-P/R/F1\nclass-wise P/R/F1'),(3.15,'Confusion matrices\nOOF counts\nwith / without TTA'),(5.8,'ROC / PR analysis\nclass-wise AUC / AP\nmicro-average'),(8.45,'Artifact robustness\nclean • hair • ruler\nocclusion'),(11.1,'XAI analysis\nnative attention vs.\nGrad-CAM')]
for x,t in outs: box(ax,x,.7,2.35,1,t)
for x in [1.7,4.325,6.975,9.625,12.275]: arr(ax,(x,2.55),(x,.7+1))
ax.text(7,.18,'Primary reported configuration: SC-EfficientNet-B4 • 384×384 • 20 epochs • Adam • lr = 1×10⁻⁴ • seed = 42',ha='center',fontsize=9)
fig.tight_layout(); fig.savefig(OUT/'Figure1_overall_pipeline_generated.png',dpi=350,bbox_inches='tight'); plt.close(fig)

print('Figure 1 regenerated:',OUT/'Figure1_overall_pipeline_generated.png')
