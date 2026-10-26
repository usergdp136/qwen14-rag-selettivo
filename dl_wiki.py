import os; os.environ['HF_HOME']='/Volumes/4TB_CRUCIAL_P310_SA/tuning/hf_cache'
from huggingface_hub import snapshot_download
for l in ['it','en']:
    p=snapshot_download('wikimedia/wikipedia',repo_type='dataset',allow_patterns=[f'20231101.{l}/*'],local_dir=f'wiki_{l}'); print(l,p,flush=True)
