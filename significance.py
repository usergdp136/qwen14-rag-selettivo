import json,random,math
from math import comb
T0='/Volumes/4TB_CRUCIAL_P310_SA/tuning'
G=json.load(open('gate.json'));top=lambda p:max(range(len(p)),key=lambda i:p[i])
def mcn(b,c):
  n=b+c; 
  if n==0: return 1.0
  k=min(b,c); return min(1,2*sum(comb(n,i) for i in range(k+1))/2**n)
def boot(d,B=10000,seed=0):
  r=random.Random(seed);n=len(d);m=sorted(sum(d[r.randrange(n)] for _ in range(n))/n for _ in range(B));return m[int(.025*B)],m[int(.975*B)]
def rep(name,a,b):
  d=[int(y)-int(x) for x,y in zip(a,b)];bb=sum(x==1 for x in d);cc=sum(x==-1 for x in d)
  lo,hi=boot(d);print(json.dumps({'set':name,'n':len(d),'norag':round(sum(a)/len(a),4),'rag':round(sum(b)/len(b),4),'delta_pt':round(100*sum(d)/len(d),2),'rag_fixes':bb,'rag_breaks':cc,'mcnemar_exact_p':round(mcn(bb,cc),4),'boot95_pt':[round(100*lo,2),round(100*hi,2)]}))
def arcsel(A,s,g):
  a=[top(r['norag'])==r['gold'] for r in A]
  b=[(top(r[s]['p']) if r[s]['rr']>=g[0] and max(r[s]['p'])>=g[1] and max(r['norag'])<g[2] else top(r['norag']))==r['gold'] for r in A];return a,b
D=json.load(open('dev_raw.json'));T=json.load(open('test_raw.json'));BASE=json.load(open(T0+'/qwen3_14b_baseline/baseline_raw.json'))
rep('ARC dev (gate tuned here, optimistic)',*arcsel(D['arc'],G['arc']['src'],G['arc']['gate']))
for s in ['wiki_it','wiki_math']: rep('ARC dev always-on '+s,[top(r['norag'])==r['gold'] for r in D['arc']],[top(r[s]['p'])==r['gold'] for r in D['arc']])
rep('ARC test gated',*arcsel(T['arc'],G['arc']['src'],G['arc']['gate']))
rep('ARC test always-on wiki_math',[top(r['norag'])==r['gold'] for r in T['arc']],[top(r['wiki_math']['p'])==r['gold'] for r in T['arc']])
for key,fld in [('gsm8k','ok')]:
  S=D[key];s=G[key]['src'];tr=G[key]['tr']
  rep('GSM8K dev gated',[r['norag'] for r in S.values()],[(r[s]['ok'] if r[s]['rr']>=tr else r['norag']) for r in S.values()])
  Bm={str(r['id']):bool(r[fld]) for r in BASE[key]['rows']};S=T[key]
  rep('GSM8K test gated',[Bm[i] for i in S],[S[i][s]['ok'] if 'ok' in S[i].get(s,{}) else Bm[i] for i in S])
