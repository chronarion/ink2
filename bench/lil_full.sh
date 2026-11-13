#!/bin/bash
# Local Inference Lab's lil-bench standard/1 (decode C1 / C8 / C16 at context 0 / 64K / 128K, prefill 32K / 128K) against
# the running inkling2 container, run inside it.   usage: lil_full.sh TAG "NOTE"
TAG=$1; LIL_SRC=${LIL_SRC:-/data/scratch/llmbench/src}; OUT=${OUT:-/data/scratch/inkling2/lil}; mkdir -p $OUT
L=$OUT/full-$TAG.log
docker exec inkling2 test -f /tmp/lilsrc/llm_decode_bench.py || { docker cp $LIL_SRC inkling2:/tmp/lilsrc >/dev/null; docker exec inkling2 sh -c 'pip install -q --target /tmp/lilpkgs httpx rich nvidia-ml-py >/dev/null 2>&1'; }
docker exec --privileged -e PYTHONPATH=/tmp/lilpkgs:/tmp/lilsrc -e LLM_BENCH_NO_UPDATE_CHECK=1 inkling2 bash -c "ulimit -n 65536; python3 -m lil_bench --no-upload --no-p2pmark --url http://127.0.0.1:8111 --note '$2'" > $L 2>&1; echo "LILEXIT $?" >> $L
sed -n '/^Results/,/PCIe/p' $L
docker logs inkling2 2>&1 | grep -E "Traceback|EngineDead|illegal memory" | head -5
