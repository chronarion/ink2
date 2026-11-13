#!/bin/bash
# quick lil-style decode A/B: same client and settings as lil-bench standard/1, chosen cells only
# usage: quick.sh TAG [CONC] [CTX]
TAG=$1; C=${2:-1,8,16}; X=${3:-0}
LIL_SRC=${LIL_SRC:-/data/scratch/llmbench/src}   # a checkout of local-inference-lab/llm-inference-bench
OUT=${OUT:-/data/scratch/inkling2/lil}; mkdir -p $OUT
docker exec inkling2 test -f /tmp/lilsrc/llm_decode_bench.py || { docker cp $LIL_SRC inkling2:/tmp/lilsrc >/dev/null; docker exec inkling2 sh -c 'pip install -q --target /tmp/lilpkgs httpx rich nvidia-ml-py >/dev/null 2>&1'; }
docker exec -e PYTHONPATH=/tmp/lilpkgs -e LLM_BENCH_NO_UPDATE_CHECK=1 -e HOME=/tmp inkling2 bash -c "ulimit -n 65536; python3 /tmp/lilsrc/llm_decode_bench.py --host http://127.0.0.1:8111 --model inkling-2-small --concurrency $C --contexts $X --duration ${DUR:-30} --max-tokens 2048 --display-mode plain --output /tmp/q-$TAG.json --no-resume --no-calibration-cache --skip-prefill --no-hw-monitor" > $OUT/q-$TAG.log 2>&1
docker cp inkling2:/tmp/q-$TAG.json $OUT/q-$TAG.json >/dev/null 2>&1
grep -A6 -E "^Aggregate decode tok/s|^MTP-normalized" $OUT/q-$TAG.log | grep -E "^│ [0-9]" | sed "s/^/$TAG /"
