# Build widened medical index chunks. Sources: wikimedia/wikipedia 20231101 (it,en; CC BY-SA 4.0) filtered by WikiProject Medicine titles; Europe PMC OA abstracts licence cc by / cc0.
# Exclusions: all cardio_v1 'heldout' articles (source of cardio test) in IT and their EN langlinks; all PubMedQA pqa_labeled PMIDs.
import json,glob,os,re,collections
import pyarrow.parquet as pq
T0='/Volumes/4TB_CRUCIAL_P310_SA/tuning'; MAXC=int(os.environ.get('MAXC','4')); CH=1500
ex=json.load(open('exclusions.json')); EXT={'it':set(ex['it_titles']),'en':set(ex['en_titles'])}; EXP=set(ex['pmids'])
PART=os.environ['PART']; out=open(f'chunks_{PART}.jsonl','w'); st=collections.Counter(); exc=collections.Counter()
def chunks(text):
    paras=[p.strip() for p in text.split('\n') if len(p.strip())>40]; cur=''
    for p in paras:
        if len(cur)+len(p)>CH and cur: yield cur; cur=''
        cur+=(p if not cur else '\n'+p)
    if cur: yield cur
for lang in (['it','en'] if PART=='wiki' else []):
    want=set(open(f'titles_{lang}.txt').read().split('\n'))
    for f in sorted(glob.glob(f'wiki_{lang}/20231101.{lang}/*.parquet')):
        t=pq.read_table(f,columns=['id','url','title','text'])
        for i,u,ti,tx in zip(*[t[c].to_pylist() for c in ['id','url','title','text']]):
            if ti not in want: continue
            if ti in EXT[lang]: exc[f'wiki_{lang}_excluded_articles']+=1; continue
            st[f'wiki_{lang}_articles']+=1
            for n,c in enumerate(chunks(tx)):
                if n>=MAXC: break
                out.write(json.dumps({'id':f'w{lang}_{i}_{n}','src':f'wiki_{lang}','title':ti,'url':u,'license':'CC BY-SA 4.0','text':c},ensure_ascii=False)+'\n'); st[f'wiki_{lang}_chunks']+=1
if PART=='wiki':  # plus rag_v1 index (cardio_v1 IT Wikipedia chunks), train split only = heldout/test sources excluded
    for c in map(json.loads,open(f'{T0}/cardio_v1/chunks.jsonl')):
        if c['split']!='train': exc['cardio_v1_heldout_chunks']+=1; continue
        out.write(json.dumps({'id':'cv1_'+c['id'],'src':'wiki_it_cardio_v1','title':c['title'],'url':c['url'],'license':'CC BY-SA 4.0','text':c['text']},ensure_ascii=False)+'\n'); st['cardio_v1_train_chunks']+=1
seen=set()
for f in (sorted(glob.glob('pmc_*.jsonl')) if PART=='pmc' else []):
    for l in open(f):
        x=json.loads(l); k=x['pmcid'] or x['pmid']
        if k in seen: continue
        seen.add(k)
        if x['pmid'] in EXP: exc['pmc_pubmedqa_excluded']+=1; continue
        ab=re.sub(r'<[^>]+>','',x['abstract']); 
        for n,c in enumerate(chunks(ab.replace('. ','.\n') if len(ab)>CH else ab)):
            if n>=2: break
            out.write(json.dumps({'id':f'pmc_{k}_{n}','src':'pmc_oa','title':x['title'],'url':f'https://europepmc.org/article/PMC/{x["pmcid"]}' if x['pmcid'] else None,'license':x['license'].upper(),'text':c},ensure_ascii=False)+'\n'); st['pmc_chunks']+=1
        st['pmc_articles']+=1; st['pmc_'+x['license'].replace(' ','_')]+=1
json.dump({'counts':st,'excluded':exc,'max_chunks_per_wiki_article':MAXC,'chunk_chars':CH},open(f'sources_{PART}.json','w'),indent=1); print(st,exc)
