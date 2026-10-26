#!/bin/sh
# resume: nohup sh run_q.sh >> run_q.log 2>&1 &
cd /Volumes/4TB_CRUCIAL_P310_SA/tuning/qwen14_rag_v1; P=../release_sft05/venv_eval/bin/python; export HF_HOME=/Volumes/4TB_CRUCIAL_P310_SA/tuning/hf_cache
while pgrep -f "run_v4.sh|run_nk.sh|run_math.sh" >/dev/null; do sleep 60; done
curl -s localhost:11434/api/generate -d '{"model":"qwen3.5:9b","keep_alive":0}' >/dev/null
echo "$(date '+%F %T') dev"; $P -u qrag.py dev | tail -2
grep -q TEST TEST_RUN_ONCE 2>/dev/null && { echo test already run; exit 0; }
echo TEST started > TEST_RUN_ONCE; echo "$(date '+%F %T') test"; $P -u qrag.py test | tail -2 && echo DONE >> TEST_RUN_ONCE
echo "$(date '+%F %T') QRAG ALL DONE"
