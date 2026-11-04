#!/bin/bash
# Reviewer cases for waybarrios/vllm-mlx#864 (lane rev-vllm-mlx-869). Observations only: no verdicts.
# usage: bash acceptance/vllm-mlx-864-rev.sh [simple|batched] [SRC]
# Oracle: HF config.json + HF tokenizer/chat template; the engine property is read from the real engine objects.

set -u
MODE="${1:-simple}"
SRC="${2:-src}"
PORT="${PORT:-18864}"
OUT="${OUT:-acceptance-out}"
HEAVY_TIMEOUT="${HEAVY_TIMEOUT:-100}"
START_DEADLINE="${START_DEADLINE:-600}"
QW="mlx-community/Qwen3-0.6B-4bit"
DS="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
GN="mlx-community/gemma-3n-E2B-it-lm-4bit"
PY="${PYTHON:-python3}"
BASE_URL="http://127.0.0.1:$PORT"
mkdir -p "$OUT/bodies" "$OUT/logs"

echo "HEADER mode=$MODE src=$SRC"
if [ -d "$SRC/.git" ]; then echo "HEADER src_sha=$(git -C "$SRC" rev-parse HEAD)"; fi
echo "HEADER $("$PY" -m pip show vllm-mlx 2>&1 | grep -E '^(Name|Version):' | tr '\n' ' ')"

cat > "$OUT/gen.py" <<'PYEOF'
import json, os, signal, sys
signal.alarm(1500)
out = os.environ["OUT"]
QW, DS, GN = os.environ["QW"], os.environ["DS"], os.environ["GN"]
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

for m in (QW, DS, GN):
    snapshot_download(m)
    with open(hf_hub_download(m, "config.json")) as f:
        c = json.load(f)
    tc = c.get("text_config") or {}
    tok = AutoTokenizer.from_pretrained(m)
    print("MODELINFO %s top_mpe=%r text_mpe=%r tok_model_max_length=%r" % (
        m, c.get("max_position_embeddings"), tc.get("max_position_embeddings"),
        tok.model_max_length))

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
toks = {m: AutoTokenizer.from_pretrained(m) for m in (QW, DS, GN)}


def ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def solve(count, target):
    lo, hi = 0, target + 64
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    if count(lo) != target:
        print("ORACLE-VOID cannot hit %d tokens exactly (got %d)" % (target, count(lo)))
        sys.exit(3)
    return lo


def chat_count(tok, mk, **kw):
    def f(n):
        text = tok.apply_chat_template(mk(n), tokenize=False, add_generation_prompt=True, **kw)
        return len(ids(tok, text))
    return f


def text_count(tok):
    return lambda n: len(ids(tok, "w " * n))


C = 40960
rows = []


def add(rid, modes, model, route, body, otok, note):
    body = dict(body)
    body["model"] = model
    with open(os.path.join(out, "bodies", rid + ".json"), "w") as f:
        json.dump(body, f)
    rows.append((rid, modes, model, route, str(otok), note))


q = toks[QW]
n = solve(text_count(q), C + 1)
long_text = "w " * n
add("q-cmpl-list3-ns", "sb", QW, "/v1/completions",
    {"prompt": ["Hello", "Hi", long_text], "max_tokens": 1, "stream": False}, C + 1,
    "completions list, third prompt over, non-stream: expect 400")
add("q-cmpl-list-first-over-stream", "s", QW, "/v1/completions",
    {"prompt": [long_text, "Hi"], "max_tokens": 1, "stream": True}, C + 1,
    "completions list stream, FIRST prompt over: expect 400")

mk_tools = lambda n: [{"role": "system", "content": "You are helpful."},
                      {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                      {"role": "user", "content": "w " * n}]
n = solve(lambda k: len(ids(q, q.apply_chat_template(mk_tools(k), tools=TOOLS, tokenize=False,
          add_generation_prompt=True, enable_thinking=True))), C + 1)
msgs = mk_tools(n)
msgs[-1] = {"role": "user", "content": [{"type": "text", "text": "w " * n}]}
add("q-chat-tools-multiturn-parts", "sb", QW, "/v1/chat/completions",
    {"messages": msgs, "tools": TOOLS, "max_tokens": 1, "stream": False}, C + 1,
    "system+multi-turn+tools+content parts, rendered 40961: expect 400")

mk_plain = lambda n: [{"role": "user", "content": "w " * n}]
n = solve(chat_count(q, mk_plain, enable_thinking=False), C + 1)
n_think = chat_count(q, mk_plain, enable_thinking=True)(n)
add("q-chat-nothink-over1", "s", QW, "/v1/chat/completions",
    {"messages": mk_plain(n), "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 1,
     "stream": False}, C + 1,
    "rendered with enable_thinking=false is 40961; default rendering is %d: expect 400" % n_think)
add("q-chat-nothink-field-over1", "s", QW, "/v1/chat/completions",
    {"messages": mk_plain(n), "enable_thinking": False, "max_tokens": 1, "stream": False}, C + 1,
    "same prompt through the enable_thinking request field: expect 400 if the field is honoured by the engine")

d = toks[DS]
n = solve(chat_count(d, mk_plain), 20000)
add("d-chat-20000", "sb", DS, "/v1/chat/completions",
    {"messages": mk_plain(n), "max_tokens": 1, "stream": False}, 20000,
    "config says 131072, tokenizer says 16384, prompt 20000")
add("d-chat-small", "sb", DS, "/v1/chat/completions",
    {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16, "stream": False}, -1,
    "small prompt control")

g = toks[GN]
n = solve(chat_count(g, mk_plain), 32769)
add("g-chat-over1", "s", GN, "/v1/chat/completions",
    {"messages": mk_plain(n), "max_tokens": 1, "stream": False}, 32769,
    "nested text_config 32768, sentinel tokenizer, prompt 32769")
add("g-chat-small", "s", GN, "/v1/chat/completions",
    {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16, "stream": False}, -1,
    "small prompt control")

with open(os.path.join(out, "rows.tsv"), "w") as f:
    for r in rows:
        f.write("\t".join(r) + "\n")
print("ORACLE ok rows=%d" % len(rows))
PYEOF

MODEL_ENV="QW=$QW DS=$DS GN=$GN OUT=$OUT"
env QW="$QW" DS="$DS" GN="$GN" OUT="$OUT" "$PY" "$OUT/gen.py" > "$OUT/gen.log" 2>&1
GEN_RC=$?
grep -E '^(MODELINFO|ORACLE)' "$OUT/gen.log" | sed 's/^/HEADER /'
if [ "$GEN_RC" -ne 0 ]; then
  echo "SUMMARY rows=0 oracle_rc=$GEN_RC"; tail -n 5 "$OUT/gen.log" | sed 's/^/GENERR /'
  exit 1
fi

cat > "$OUT/limit_probe.py" <<'PYEOF'
import asyncio, os, sys, traceback
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
        lim = getattr(e, "context_limit", "ATTRIBUTE-ABSENT")
        print("LIMIT mode=%s model=%s is_mllm=%r context_limit=%r tokenizer_mml=%r" % (
            mode, model, getattr(e, "_is_mllm", None), lim,
            getattr(getattr(e.tokenizer, "_tokenizer", e.tokenizer), "model_max_length", None)))
        cfgs = []
        m = getattr(e, "_model", None)
        inner = getattr(m, "model", m)
        for a in ("config", "args", "model_config"):
            v = getattr(inner, a, None)
            cfgs.append("%s=%s" % (a, type(v).__name__ if v is not None else None))
        print("LIMIT-ATTRS " + " ".join(cfgs))
    finally:
        await e.stop()
try:
    asyncio.run(main())
except BaseException:
    print("LIMIT-ERROR mode=%s model=%s %s" % (mode, model, traceback.format_exc().strip().splitlines()[-1]))
PYEOF
for m in "$QW" "$DS" "$GN"; do
  timeout 600 "$PY" "$OUT/limit_probe.py" "$MODE" "$m" > "$OUT/logs/probe-$MODE.log" 2>&1 < /dev/null
  grep -E '^(LIMIT|LIMIT-ATTRS|LIMIT-ERROR)' "$OUT/logs/probe-$MODE.log" | sed "s/^/PROBE /"
  grep -q '^LIMIT' "$OUT/logs/probe-$MODE.log" || { echo "PROBE no-output model=$m"; tail -n 3 "$OUT/logs/probe-$MODE.log" | sed 's/^/PROBEERR /'; }
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

flag="s"; [ "$MODE" = "batched" ] && flag="b"
N=0
while IFS="$(printf '\t')" read -r id modes model route otok note <&3; do
  case "$modes" in *"$flag"*) ;; *) continue ;; esac
  N=$((N+1))
  slog="$OUT/logs/server-$MODE-$id.log"
  if ! start_server "$model" "$slog"; then
    echo "RESULT mode=$MODE id=$id model=$model route=$route oracle_prompt_tokens=$otok http=000 curl_rc=99 elapsed=0 note=\"$note\" body=\"$START_MSG\""
    stop_server; continue
  fi
  bodyf="$OUT/resp-$MODE-$id.body"; : > "$bodyf"
  meta=$(curl -sS -N -m "$HEAVY_TIMEOUT" -o "$bodyf" -w '%{http_code} %{time_total}' \
    -X POST "$BASE_URL$route" -H 'content-type: application/json' \
    --data-binary "@$OUT/bodies/$id.json" 2>"$OUT/resp-$MODE-$id.err")
  crc=$?
  http=$(echo "$meta" | awk '{print $1}'); elapsed=$(echo "$meta" | awk '{print $2}')
  [ -z "$http" ] && http="000"; [ -z "$elapsed" ] && elapsed="$HEAVY_TIMEOUT"
  stop_server
  snippet=$(head -c 260 "$bodyf" | tr '\n\r' '  ' | tr -d '"')
  echo "RESULT mode=$MODE id=$id model=$model route=$route oracle_prompt_tokens=$otok http=$http curl_rc=$crc elapsed=$elapsed note=\"$note\" body=\"$snippet\""
done 3< "$OUT/rows.tsv"
echo "SUMMARY mode=$MODE rows=$N"
exit 0
