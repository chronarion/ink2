#!/bin/bash
# ink2: Inkling 2 Small on one DGX Station GB300 (vLLM's Inkling 2 model; routed experts split between GPU memory and
# Grace memory, ink2/split.py).
#   ink2.sh up | down | status | logs
#   PROFILE=fast ink2.sh up: also the RTX PRO 6000 (3-bit copies of the warm experts), GPTQ NVFP4 layers 40-47 and a
#   larger hot set, from FAST_DIR (the "fast" folder of huggingface.co/chronarion/Inkling-2-Small-ink2)
#   ENTRY / ENTRY_ARGS (development): another entrypoint wrapping vllm, e.g. ENTRY=/nsys/bin/nsys
#   ENTRY_ARGS="profile ... -o /prof/run vllm" with DOCKER_EXTRA mounting /opt/nvidia/nsight-systems at /nsys
set -euo pipefail
E=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME=${NAME:-inkling2}
PORT=${PORT:-8111}
IMG=${IMG:-vllm/vllm-openai:inkling-2-small-arm64}
MODEL=${MODEL:-$(ls -d /data/hf/hub/models--thinkingmachines--Inkling-2-Small/snapshots/*/ 2>/dev/null | head -1)}
GB=${GB:-GPU-73779fc7-ce41-b87b-5239-16e36ce490fe}
RTX=${RTX:-GPU-84b21d5a-e28c-3068-93f1-5ca038d90ded}
PROFILE=${PROFILE:-standard}
if [ "$PROFILE" = fast ]; then
  FAST_DIR=${FAST_DIR:-$(ls -d /data/hf/hub/models--chronarion--Inkling-2-Small-ink2/snapshots/*/fast 2>/dev/null | head -1)}
  if [ ! -d "${FAST_DIR:-}/band" ] || [ ! -d "$FAST_DIR/gptq40" ] || [ ! -f "$FAST_DIR/placement.json" ]; then
    echo "PROFILE=fast needs FAST_DIR with band/, gptq40/ and placement.json (got '${FAST_DIR:-}'; see README)"; exit 2
  fi
  FAST_DIR=$(cd "$FAST_DIR" && pwd)
  DOCKER_EXTRA="${DOCKER_EXTRA:-} -v $FAST_DIR:$FAST_DIR:ro"
  export INK2_BAND=${INK2_BAND:-$FAST_DIR/band} INK2_EXPERT_OVERRIDE=${INK2_EXPERT_OVERRIDE:-$FAST_DIR/gptq40}
  # DSpark (the draft model in the checkpoint) is on: 3 drafts at 1 request, 1 at 2-16, none above (quick lil cells, 11-14:
  # 197 / 723 / 995 vs 181 / 619 / 945 with 2 / 1 (2-4) / 0); its placement leaves GPU memory for the drafter.
  # SPEC= (empty) turns it off; while it is on, vLLM refuses min_p and logit_bias.
  SPEC=${SPEC-'{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","num_speculative_tokens_per_batch_size":[[1,1,3],[2,16,1],[17,32,0]]}'}
  if [ -n "$SPEC" ]; then FAST_PLACEMENT=placement-dspark.json; else FAST_PLACEMENT=placement.json; fi
  export INK2_PLACEMENT=${INK2_PLACEMENT:-$FAST_DIR/$FAST_PLACEMENT}
elif [ "$PROFILE" != standard ]; then
  echo "PROFILE is standard (default) or fast"; exit 2
fi
# INK2_BAND (dir of TR3 band weights, ink2/band.py): the RTX joins the container as cuda:1 (GB300 first); its kernels
# build from RTX_SRC (the moe3i sources) with the CUDA headers the image lacks (inc/, symlinks into the image)
RTX_SRC=${RTX_SRC:-$E/rtx}
GPUS="--device nvidia.com/gpu=$GB"
# INK2_GRACE (ink2/graceband.py, experimental, off): the tail on the Grace cores; its host code builds from cpuband/
if [ -n "${INK2_GRACE:-}" ]; then
  GPUS="$GPUS -v $(cd ${CPUBAND_SRC:-$E/cpuband} && pwd):/ink2cpuband:ro"
fi
if [ -n "${INK2_BAND:-}" ]; then
  GPUS="$GPUS --device nvidia.com/gpu=$RTX -e CUDA_VISIBLE_DEVICES=$GB,$RTX -v $(cd $RTX_SRC && pwd):/ink2rtx:ro -e CPATH=/ink2engine/inc"
fi
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
  log "starting: profile $PROFILE${SPEC:+ + speculative decoding}, port $PORT, max model length $MAXLEN, placement ${INK2_PLACEMENT:-none (INK2_HOT ${INK2_HOT:-192})}"
  docker run -d --name $NAME --restart no $GPUS --ipc host --network host \
    --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --security-opt label=disable \
    -v /data/hf:/data/hf:ro -v $E:/ink2engine:ro -e PYTHONPATH=/ink2engine ${DOCKER_EXTRA:-} \
    -v $STATE/vllm:/root/.cache/vllm -v $STATE/triton:/root/.triton -v $STATE/flashinfer:/root/.cache/flashinfer \
    $(env | grep -o "^INK2_[A-Z0-9_]*" | sed "s/^/-e /" | tr "\n" " ") \
    -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO \
    --entrypoint ${ENTRY:-vllm} $IMG ${ENTRY_ARGS:-} serve $MODEL --served-model-name inkling-2-small Inkling-2-Small inkling2 \
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
