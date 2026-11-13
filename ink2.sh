#!/bin/bash
# ink2: Inkling 2 Small on one DGX Station GB300 (vLLM's Inkling 2 model; routed experts split between GPU memory and
# Grace memory, ink2/split.py).
#   ink2.sh up | down | status | logs
set -euo pipefail
E=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME=${NAME:-inkling2}
PORT=${PORT:-8111}
IMG=${IMG:-vllm/vllm-openai:inkling-2-small-arm64}
MODEL=${MODEL:-$(ls -d /data/hf/hub/models--thinkingmachines--Inkling-2-Small/snapshots/*/ 2>/dev/null | head -1)}
GB=${GB:-GPU-73779fc7-ce41-b87b-5239-16e36ce490fe}
MAXLEN=${MAXLEN:-1048576}
# expert placement (engine/placement/*.json, built by tools/mkplace.py from decode routing counts)
export INK2_PLACEMENT=${INK2_PLACEMENT:-/ink2engine/placement/gen-count167-s64.json}
# routing counts the placement was built from: the promotion's prior (ink2/follow.py)
export INK2_FOLLOW_PRIOR=${INK2_FOLLOW_PRIOR:-/ink2engine/placement/counts-gen.json}
STATE=${STATE:-$E/state}
log() { echo "[ink2 $(date +%T)] $*"; }

down() { docker rm -f $NAME >/dev/null 2>&1 || true; log down; }

up() {
  down
  mkdir -p $STATE/vllm $STATE/triton $STATE/flashinfer
  log "starting: port $PORT, max model length $MAXLEN, placement ${INK2_PLACEMENT:-none (INK2_HOT ${INK2_HOT:-192})}"
  docker run -d --name $NAME --restart no --device nvidia.com/gpu=$GB --ipc host --network host \
    --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --security-opt label=disable \
    -v /data/hf:/data/hf:ro -v $E:/ink2engine:ro -e PYTHONPATH=/ink2engine ${DOCKER_EXTRA:-} \
    -v $STATE/vllm:/root/.cache/vllm -v $STATE/triton:/root/.triton -v $STATE/flashinfer:/root/.cache/flashinfer \
    $(env | grep -o "^INK2_[A-Z0-9_]*" | sed "s/^/-e /" | tr "\n" " ") \
    -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO \
    --entrypoint vllm $IMG serve $MODEL --served-model-name inkling-2-small Inkling-2-Small inkling2 \
    --tensor-parallel-size 1 --max-model-len $MAXLEN --max-num-seqs ${MAXSEQS:-32} --max-num-batched-tokens ${MAXBT:-4096} \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.94} \
    --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
    ${SPEC:+--speculative-config "$SPEC"} ${EXTRA_ARGS:-} \
    --host 0.0.0.0 --port $PORT >/dev/null
  for i in $(seq 1 240); do
    st=$(docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null || echo missing)
    [ "$st" != running ] && { log "container $st"; docker logs --tail 40 $NAME 2>&1; return 1; }
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then
      log "ready on :$PORT after $((i*15)) s"; return 0
    fi
    sleep 15
  done
  log "not ready after 60 min"; return 1
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) docker ps --filter name=$NAME --format '{{.Names}} {{.Status}}'; curl -s http://127.0.0.1:$PORT/v1/models | head -c 300; echo ;;
  logs) docker logs --tail 80 $NAME ;;
  *) echo "usage: $0 up | down | status | logs"; exit 2 ;;
esac
