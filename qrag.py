# qwen14_rag_v1: Qwen3-14B BASE (mlx q8, non-thinking, greedy; same code paths as qwen3_14b_baseline/bench_qwen.py) vs base+RAG.
# Sources tried separately: wiki_it (rag_v2 chunks, CC BY-SA) and wiki_math (math_rag_v1, CC BY-SA); BM25 top-3, rr = normalised top score.
# DEV (disjoint): ARC-it validation (all), GSM8K train 60, IFEval 60 not in the 100 test ids. Gate per task tuned on DEV; source kept only if dev net>0.
# TEST once: ARC-it full test 1132 (norag+rag), GSM8K 150 / IFEval 100 same ids as baseline (norag rows reused from baseline_raw.json, rag only where gate fires).
import sys,json,random,re,time,os,math,itertools
from collections import Counter
import mlx.core as mx
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
from datasets import load_dataset
T0='/Volumes/4TB_CRUCIAL_P310_SA/tuning'; stage=sys.argv[1]
M=T0+'/hf_cache/hub/models--mlx-community--Qwen3-14B-8bit/snapshots/da33cf28f06636847fd9e93e0a03d819b84cb55e'
tokz=lambda s: re.findall(r'\w{2,}',s.lower())
class BM:
    def __init__(s,C):
        s.C=C; s.TF=[Counter(tokz(c['title']+' '+c['text'])) for c in C]; s.L=[sum(t.values()) for t in s.TF]; s.avg=sum(s.L)/len(s.L); s.N=len(C)
        s.DF=Counter(w for t in s.TF for w in t); s.inv={}
        for n,t in enumerate(s.TF):
            for w in t: s.inv.setdefault(w,[]).append(n)
    def __call__(s,q,k=3):
        qs=[w for w in set(tokz(q)) if w in s.DF]; idf={w:math.log(1+(s.N-s.DF[w]+.5)/(s.DF[w]+.5)) for w in qs}; sc=Counter()
        for w in qs:
            for n in s.inv[w]: tf=s.TF[n][w]; sc[n]+=idf[w]*tf*2.2/(tf+1.2*(.25+.75*s.L[n]/s.avg))
        mxs=sum(idf.values()) or 1; return [(round(v/mxs,4),n) for n,v in sc.most_common(k)]
SRC={'wiki_it':[c for c in map(json.loads,open(T0+'/rag_v2/chunks_wiki.jsonl')) if c.get('src')=='wiki_it'],'wiki_math':[json.loads(l) for l in open(T0+'/math_rag_v1/chunks.jsonl')]}
IDX={k:BM(v) for k,v in SRC.items()}; print({k:len(v) for k,v in SRC.items()},flush=True)
H="Seguono brani di riferimento. Usali SOLO se sono direttamente utili per rispondere; altrimenti ignorali.\n\nBRANI:\n"
def blk(src,P): C=SRC[src]; return H+'\n\n'.join('[%d] %s\n%s'%(i+1,C[n]['title'],C[n]['text'][:700]) for i,(s,n) in enumerate(P))+'\n\nDOMANDA:\n'
mx.set_default_device(mx.cpu); m,tok=load(M); mx.eval(m.parameters()); mx.set_default_device(mx.gpu); smp=make_sampler(temp=0.0)
chat=lambda u: tok.apply_chat_template([{'role':'user','content':u}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
gen=lambda u,n: generate(m,tok,prompt=chat(u),max_tokens=n,sampler=smp,verbose=False)
L='ABCDE'; lid=[tok.encode(' '+c,add_special_tokens=False)[-1] for c in L]; lid2=[tok.encode(c,add_special_tokens=False)[-1] for c in L]
def arcp(q,k):
    lg=m(mx.array(tok.encode(chat(q),add_special_tokens=False))[None])[0,-1].astype(mx.float32); sc=[max(float(lg[lid[i]]),float(lg[lid2[i]])) for i in range(k)]
    z=max(sc); e=[math.exp(x-z) for x in sc]; return [x/sum(e) for x in e]
def gsm_ok(o,gold):
    nums=re.findall(r'-?\d[\d,]*\.?\d*',o.split('answer is')[-1] if 'answer is' in o else o)
    try: p=float(nums[-1].replace(',','').rstrip('.'))
    except Exception: return False
    return abs(p-gold)<1e-6
from lm_eval.tasks.ifeval import utils as U
OUT=stage+'_raw.json'; R=json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R,open(OUT+'.tmp','w')); os.replace(OUT+'.tmp',OUT)
G=json.load(open('gate.json')) if stage=='test' else None
BASE=json.load(open(T0+'/qwen3_14b_baseline/baseline_raw.json'))
# ARC
arc=load_dataset('sapienzanlp/arc_italian','arc_challenge',split='validation' if stage=='dev' else 'test'); A=R.setdefault('arc',[])
for i in range(len(A),len(arc)):
    d=arc[i]; ch=d['choices_translation']; k=len(ch); q=d['input_translation']+'\n'+'\n'.join(f'{L[j]}. {c}' for j,c in enumerate(ch))+'\nRispondi solo con la lettera della risposta corretta.'
    r={'gold':d['label'],'norag':arcp(q,k)}
    for s in (SRC if stage=='dev' else ([G['arc']['src']] if G['arc']['keep'] else [])):
        P=IDX[s](d['input_translation']+' '+' '.join(ch)); r[s]={'rr':P[0][0] if P else 0,'p':arcp(blk(s,P)+q,k) if P else r['norag']}
    A.append(r)
    if i%25==0: save(); print('arc',i,flush=True)
save()
def gen_task(key,items,run,gold):
    S=R.setdefault(key,{})
    for it in items:
        if str(it) in S: continue
        q=run('q',it); r={}
        r['norag']=run('norag',it) if stage=='dev' else None
        srcs=list(SRC) if stage=='dev' else ([G[key]['src']] if G[key]['keep'] else [])
        for s in srcs:
            P=IDX[s](q); rr=P[0][0] if P else 0
            if stage=='dev' or rr>=G[key]['tr']: r[s]={'rr':rr,'ok':run(('rag',s,P),it)}
            else: r[s]={'rr':rr}
        S[str(it)]=r; save(); print(key,len(S),flush=True)
if stage=='dev':
    gs=load_dataset('openai/gsm8k','main',split='train'); gi=sorted(random.Random(99).sample(range(len(gs)),60))
else:
    gs=load_dataset('openai/gsm8k','main',split='test'); gi=[r['id'] for r in BASE['gsm8k']['rows']]
def grun(mode,i):
    d=gs[i]; q=d['question']
    if mode=='q': return q
    gold=float(d['answer'].split('####')[-1].replace(',','').strip()); pr=q+"\nSolve step by step, then write the final answer on the last line as 'The answer is N.'"
    if mode!='norag': pr=blk(mode[1],mode[2]).replace('DOMANDA','PROBLEM')+pr
    return gsm_ok(gen(pr,512),gold)
gen_task('gsm8k',gi,grun,None)
ifd=load_dataset('google/IFEval',split='train'); tid=[r['id'] for r in BASE['ifeval']['rows']]
ii=sorted(random.Random(99).sample([i for i in range(len(ifd)) if i not in set(tid)],60)) if stage=='dev' else tid
def irun(mode,i):
    d=ifd[i]
    if mode=='q': return d['prompt']
    pr=d['prompt'] if mode=='norag' else blk(mode[1],mode[2])+d['prompt']
    return bool(U.process_results(d,[gen(pr,1024)])['prompt_level_strict_acc'])
gen_task('ifeval',ii,irun,None)
mxp=lambda p: max(p)
top=lambda p: max(range(len(p)),key=lambda i:p[i])
if stage=='dev':
    G={}
    best=None
    for s in SRC:
        for g in itertools.product([0,.2,.3,.4,.5,.6,.7],[0,.5,.7,.9],[.5,.7,.9,1.01]):
            fin=[top(r[s]['p']) if r[s]['rr']>=g[0] and mxp(r[s]['p'])>=g[1] and mxp(r['norag'])<g[2] else top(r['norag']) for r in A]
            h=sum(f==r['gold'] and top(r['norag'])!=r['gold'] for f,r in zip(fin,A)); u=sum(top(r['norag'])==r['gold'] and f!=r['gold'] for f,r in zip(fin,A))
            key=(h-u,-u,g[0])
            if best is None or key>best[0]: best=(key,s,g)
    G['arc']={'src':best[1],'gate':best[2],'net':best[0][0],'keep':best[0][0]>0,'norag':round(sum(top(r['norag'])==r['gold'] for r in A)/len(A),4),
              'always_on':{s:round(sum(top(r[s]['p'])==r['gold'] for r in A)/len(A),4) for s in SRC}}
    for key in ['gsm8k','ifeval']:
        S=R[key]; b=(0,None,9)
        for s in SRC:
            for tr in sorted({r[s]['rr'] for r in S.values()}):
                net=sum(r[s]['ok']-r['norag'] for r in S.values() if r[s]['rr']>=tr)
                if net>b[0]: b=(net,s,tr)
        G[key]={'src':b[1],'tr':b[2],'net':b[0],'keep':b[0]>0,'norag':round(sum(r['norag'] for r in S.values())/len(S),4),'always_on':{s:round(sum(r[s]['ok'] for r in S.values())/len(S),4) for s in SRC}}
    json.dump(G,open('gate.json','w'),indent=1); print(json.dumps(G))
else:
    out={}
    def fa(r):
        if not G['arc']['keep']: return top(r['norag'])
        s=G['arc']['src']; g=G['arc']['gate']; x=r[s]
        return top(x['p']) if x['rr']>=g[0] and mxp(x['p'])>=g[1] and mxp(r['norag'])<g[2] else top(r['norag'])
    out['arc_it']={'norag':round(sum(top(r['norag'])==r['gold'] for r in A)/len(A),4),'rag':round(sum(fa(r)==r['gold'] for r in A)/len(A),4),'n':len(A),'baseline_ref':BASE['arc_it']['acc']}
    for key,bk,fld in [('gsm8k','gsm8k','ok'),('ifeval','ifeval','prompt_level_strict_acc')]:
        Bm={str(r['id']):bool(r[fld]) for r in BASE[bk]['rows']}; S=R[key]; s=G[key]['src']
        fin=[S[i][s]['ok'] if (G[key]['keep'] and 'ok' in S[i].get(s,{})) else Bm[i] for i in S]
        out[key]={'norag':round(sum(Bm.values())/len(Bm),4),'rag':round(sum(fin)/len(fin),4),'n':len(fin),'rag_used':sum(G[key]['keep'] and 'ok' in S[i].get(s,{}) for i in S)}
    out['success']=all(out[k]['rag']>=out[k]['norag'] for k in ['arc_it','gsm8k','ifeval']) and any(out[k]['rag']>out[k]['norag'] for k in ['arc_it','gsm8k','ifeval'])
    out['gate']=G; json.dump(out,open('test_summary.json','w'),indent=1); print(json.dumps(out))
print('QRAG',stage,'DONE')
