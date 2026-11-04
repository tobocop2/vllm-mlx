#!/bin/bash
# Reviewer cases for waybarrios/vllm-mlx#869 (lane rev-vllm-mlx-869, delta round): a model whose
# max_position_embeddings sits under text_config, served in text mode, against a flat control.
# usage: bash acceptance/vllm-mlx-864-nested.sh [simple|batched] [SRC]
# Both models are copied to local directories with the declared context set to 2048, so a prompt
# over the declared context is small enough to run. The directory names match no MLLM pattern and
# the multimodal config keys are removed from the nested copy, so the text route is taken.
set -u
MODE="${1:-simple}"
SRC="${2:-src}"
PORT="${PORT:-18864}"
OUT="${OUT:-acceptance-out}"
MODELS="$OUT.models"
HEAVY_TIMEOUT="${HEAVY_TIMEOUT:-100}"
START_DEADLINE="${START_DEADLINE:-420}"
PY="${PYTHON:-python3}"
BASE_URL="http://127.0.0.1:$PORT"
mkdir -p "$OUT/bodies" "$OUT/logs" "$MODELS"
echo "HEADER mode=$MODE src=$SRC"
if [ -d "$SRC/.git" ]; then echo "HEADER src_sha=$(git -C "$SRC" rev-parse HEAD)"; fi

cat > "$OUT/gen.py" <<'PYEOF'
import json, os, shutil, signal, sys
signal.alarm(1500)
out, models = os.environ["OUT"], os.environ["MODELS"]
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

def build(repo, name, edit):
    src = snapshot_download(repo)
    dst = os.path.join(models, name)
    if not os.path.isdir(dst):
        shutil.copytree(src, dst, symlinks=False)
    p = os.path.join(dst, "config.json")
    c = json.load(open(p))
    edit(c)
    json.dump(c, open(p, "w"))
    return dst

def edit_flat(c):
    c["max_position_embeddings"] = 2048

def edit_nested(c):
    for k in ("vision_config", "audio_config", "vision_tower", "mm_vision_tower", "image_token_id", "image_token_index", "audio_token_id", "audio_token_index"):
        c.pop(k, None)
    c["text_config"]["max_position_embeddings"] = 2048

flat = build("mlx-community/Qwen3-0.6B-4bit", "flatlm", edit_flat)
nested = build("mlx-community/Qwen3.5-0.8B-4bit", "nestedlm", edit_nested)
rows = []
for name, d in (("flat", flat), ("nested", nested)):
    c = json.load(open(os.path.join(d, "config.json")))
    tc = c.get("text_config") or {}
    tok = AutoTokenizer.from_pretrained(d)
    print("MODELINFO %s dir=%s model_type=%s archs=%s top_mpe=%r text_mpe=%r tok_model_max_length=%r" % (
        name, d, c.get("model_type"), c.get("architectures"), c.get("max_position_embeddings"), tc.get("max_position_embeddings"), tok.model_max_length))
    for kind, n in (("small", 8), ("over", 2300)):
        msgs = [{"role": "user", "content": "w " * n}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        count = len(tok(text, add_special_tokens=False)["input_ids"])
        rid = "%s-%s" % (name, kind)
        body = {"model": d, "messages": msgs, "max_tokens": 4, "stream": False}
        with open(os.path.join(out, "bodies", rid + ".json"), "w") as f:
            json.dump(body, f)
        rows.append((rid, d, str(count)))
with open(os.path.join(out, "rows.tsv"), "w") as f:
    for r in rows:
        f.write("\t".join(r) + "\n")
print("ORACLE ok rows=%d" % len(rows))
PYEOF
env OUT="$OUT" MODELS="$MODELS" "$PY" "$OUT/gen.py" > "$OUT/gen.log" 2>&1
GEN_RC=$?
grep -E '^(MODELINFO|ORACLE)' "$OUT/gen.log" | sed 's/^/HEADER /'
if [ "$GEN_RC" -ne 0 ]; then echo "SUMMARY rows=0 oracle_rc=$GEN_RC"; tail -n 5 "$OUT/gen.log" | sed 's/^/GENERR /'; exit 1; fi

cat > "$OUT/limit_probe.py" <<'PYEOF'
import asyncio, signal, sys, traceback
signal.alarm(600)
mode, model = sys.argv[1], sys.argv[2]
async def main():
    if mode == "batched":
        from vllm_mlx.engine.batched import BatchedEngine
        e = BatchedEngine(model)
    else:
        from vllm_mlx.engine.simple import SimpleEngine
        e = SimpleEngine(model)
    await e.start()
    try:
        inner = getattr(getattr(e, "_model", None), "model", getattr(e, "_model", None))
        attrs = []
        for a in ("config", "args", "model_config"):
            v = getattr(inner, a, None)
            attrs.append("%s=%s" % (a, type(v).__name__ if v is not None else None))
        print("LIMIT mode=%s model=%s is_mllm=%r context_limit=%r attrs=[%s]" % (
            mode, model, getattr(e, "_is_mllm", None), getattr(e, "context_limit", "ABSENT"), " ".join(attrs)))
    finally:
        await e.stop()
try:
    asyncio.run(main())
except BaseException:
    print("LIMIT-ERROR mode=%s model=%s %s" % (mode, model, traceback.format_exc().strip().splitlines()[-1]))
PYEOF
for d in flatlm nestedlm; do
  "$PY" "$OUT/limit_probe.py" "$MODE" "$MODELS/$d" > "$OUT/logs/probe-$MODE-$d.log" 2>&1 < /dev/null
  grep -E '^(LIMIT|LIMIT-ERROR)' "$OUT/logs/probe-$MODE-$d.log" | sed "s/^/PROBE /"
  grep -q '^LIMIT' "$OUT/logs/probe-$MODE-$d.log" || { echo "PROBE no-output dir=$d"; tail -n 3 "$OUT/logs/probe-$MODE-$d.log" | sed 's/^/PROBEERR /'; }
done

SERVER_PID=""
start_server() {
  model="$1"; log="$2"
  extra=""
  if [ "$MODE" = "batched" ]; then extra="--continuous-batching"; fi
  nohup vllm-mlx serve "$model" --host 127.0.0.1 --port "$PORT" $extra > "$log" 2>&1 < /dev/null &
  SERVER_PID=$!
  t0=$(date +%s)
  while :; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then START_MSG="server-exited-early"; return 1; fi
    h=$(curl -s -m 5 "$BASE_URL/health" 2>/dev/null || true)
    case "$h" in *'"model_loaded":true'*) START_MSG=""; return 0 ;; esac
    if [ $(( $(date +%s) - t0 )) -gt "$START_DEADLINE" ]; then START_MSG="server-start-deadline"; return 1; fi
    sleep 3
  done
}
stop_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    n=0
    while kill -0 "$SERVER_PID" 2>/dev/null && [ "$n" -lt 20 ]; do sleep 1; n=$((n+1)); done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT

N=0
while IFS="$(printf '\t')" read -r id model otok <&3; do
  N=$((N+1))
  slog="$OUT/logs/server-$MODE-$id.log"
  if ! start_server "$model" "$slog"; then
    echo "RESULT mode=$MODE id=$id oracle_prompt_tokens=$otok http=000 curl_rc=99 elapsed=0 body=\"$START_MSG\""
    stop_server; continue
  fi
  bodyf="$OUT/resp-$MODE-$id.body"; : > "$bodyf"
  meta=$(curl -sS -N -m "$HEAVY_TIMEOUT" -o "$bodyf" -w '%{http_code} %{time_total}' \
    -X POST "$BASE_URL/v1/chat/completions" -H 'content-type: application/json' \
    --data-binary "@$OUT/bodies/$id.json" 2>"$OUT/resp-$MODE-$id.err")
  crc=$?
  http=$(echo "$meta" | awk '{print $1}'); elapsed=$(echo "$meta" | awk '{print $2}')
  [ -z "$http" ] && http="000"; [ -z "$elapsed" ] && elapsed="$HEAVY_TIMEOUT"
  stop_server
  snippet=$(head -c 300 "$bodyf" | tr '\n\r' '  ' | tr -d '"')
  echo "RESULT mode=$MODE id=$id oracle_prompt_tokens=$otok http=$http curl_rc=$crc elapsed=$elapsed body=\"$snippet\""
done 3< "$OUT/rows.tsv"
echo "SUMMARY mode=$MODE rows=$N"
exit 0
