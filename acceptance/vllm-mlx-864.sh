#!/bin/bash
# Acceptance script for waybarrios/vllm-mlx#864: a prompt longer than the model
# context must get a fast client error that names the limit.
#
# usage: bash acceptance/vllm-mlx-864.sh [simple|batched|all] [SRC]
#   simple   SimpleEngine   (`vllm-mlx serve MODEL`)
#   batched  BatchedEngine  (`vllm-mlx serve MODEL --continuous-batching`)
#   SRC      checkout under test (default: src); only used to find the log tag
#
# env: MODEL (default mlx-community/Qwen3-0.6B-4bit), PORT (default 18864),
#      OUT (default acceptance-out), BASE_FILE (result of the base run, used by
#      KEEP rows), HEAVY_TIMEOUT (default 100 s), SMALL_TIMEOUT (default 60 s),
#      OVER_DEADLINE (default 10 s: the longest an OVER row may take).
#
# The oracle is not the code under test: the context length is read from the
# model's config.json (huggingface_hub) and the prompt length is counted with the
# HF tokenizer and chat template (transformers). No vllm_mlx code is imported.
#
# Output: one RESULT line per cell, then SUMMARY. Exit 0 only when every cell
# passes and every expected cell ran.

set -u

MODE_ARG="${1:-all}"
SRC="${2:-src}"
MODEL="${MODEL:-mlx-community/Qwen3-0.6B-4bit}"
PORT="${PORT:-18864}"
OUT="${OUT:-acceptance-out}"
BASE_FILE="${BASE_FILE:-acceptance/vllm-mlx-864.base.txt}"
HEAVY_TIMEOUT="${HEAVY_TIMEOUT:-100}"
SMALL_TIMEOUT="${SMALL_TIMEOUT:-60}"
OVER_DEADLINE="${OVER_DEADLINE:-10}"
START_DEADLINE="${START_DEADLINE:-420}"
EXPECTED_CONTEXT=40960

case "$MODE_ARG" in
  simple) MODES="simple" ;;
  batched) MODES="batched" ;;
  all) MODES="simple batched" ;;
  *) echo "usage: $0 [simple|batched|all] [SRC]"; exit 2 ;;
esac

mkdir -p "$OUT/bodies" "$OUT/logs"
PY="${PYTHON:-python3}"
BASE_URL="http://127.0.0.1:$PORT"

echo "HEADER src=$SRC model=$MODEL modes=\"$MODES\" heavy_timeout=${HEAVY_TIMEOUT}s small_timeout=${SMALL_TIMEOUT}s over_deadline=${OVER_DEADLINE}s"
if [ -d "$SRC/.git" ]; then
  echo "HEADER src_sha=$(git -C "$SRC" rev-parse HEAD)"
fi
echo "HEADER $("$PY" -m pip show vllm-mlx 2>&1 | grep -E '^(Name|Version):' | tr '\n' ' ')"

# ---------------------------------------------------------------- generator + oracle
cat > "$OUT/gen.py" <<'PYEOF'
import json, os, signal, sys

signal.alarm(900)
model = os.environ["MODEL"]
out = os.environ["OUT"]
expected_ctx = int(os.environ["EXPECTED_CONTEXT"])

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

with open(hf_hub_download(model, "config.json")) as f:
    cfg = json.load(f)
ctx = cfg.get("max_position_embeddings")
if ctx != expected_ctx:
    print("ORACLE-VOID max_position_embeddings=%r expected %d" % (ctx, expected_ctx))
    sys.exit(3)

tok = AutoTokenizer.from_pretrained(model)


def ids(text):
    return tok(text, add_special_tokens=False)["input_ids"]


def chat_tokens(n):
    msg = [{"role": "user", "content": "w " * n}]
    text = tok.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    return len(ids(text))


def text_tokens(n):
    return len(ids("w " * n))


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


SMALL_CHAT = "Say hi."
SMALL_TEXT = "Hello"
C = ctx
# id, modes (s=simple b=batched), kind, stream, class, target tokens (None=small), max_tokens
ROWS = [
    ("chat-small-ns", "sb", "chat", False, "SMALL", None, 16),
    ("chat-small-stream", "sb", "chat", True, "SMALL", None, 16),
    ("chat-under-ns", "sb", "chat", False, "FITS", C - 1, 1),
    ("chat-under-stream", "s", "chat", True, "FITS", C - 1, 1),
    ("chat-at-ns", "s", "chat", False, "KEEP", C, 1),
    ("chat-sumover-ns", "s", "chat", False, "KEEP", C - 8, 16),
    ("chat-over1-ns", "sb", "chat", False, "OVER", C + 1, 1),
    ("chat-over1-stream", "sb", "chat", True, "OVER", C + 1, 1),
    ("chat-repro-ns", "sb", "chat", False, "OVER", 60008, 16),
    ("chat-repro-stream", "s", "chat", True, "OVER", 60008, 16),
    ("chat-maxtok-ns", "sb", "chat", False, "MAXTOK", None, 40000),
    ("chat-both-ns", "sb", "chat", False, "MAXTOK", C + 1, 40000),
    ("cmpl-small-ns", "sb", "text", False, "SMALL", None, 16),
    ("cmpl-small-stream", "s", "text", True, "SMALL", None, 16),
    ("cmpl-over1-ns", "sb", "text", False, "OVER", C + 1, 1),
    ("cmpl-over1-stream", "s", "text", True, "OVER", C + 1, 1),
    ("cmpl-list-ns", "s", "textlist", False, "OVER", C + 1, 1),
    ("cmpl-list-stream", "s", "textlist", True, "KEEP", C + 1, 1),
    ("cmpl-maxtok-ns", "s", "text", False, "MAXTOK", None, 40000),
    ("anth-small", "s", "anthropic", False, "SMALL", None, 16),
    ("anth-over1", "s", "anthropic", False, "KEEP", C + 1, 1),
    ("resp-small", "s", "responses", False, "SMALL", None, 16),
    ("resp-over1", "s", "responses", False, "KEEP", C + 1, 1),
]

cache = {}


def n_for(kind, target):
    k = ("text" if kind in ("text", "textlist") else "chat", target)
    if k not in cache:
        cache[k] = solve(text_tokens if k[0] == "text" else chat_tokens, target)
    return cache[k]


rows_out = []
for rid, modes, kind, stream, cls, target, max_tokens in ROWS:
    if target is None:
        chat_text, text_text, oracle_tokens = SMALL_CHAT, SMALL_TEXT, -1
    else:
        n = n_for(kind, target)
        chat_text = text_text = "w " * n
        oracle_tokens = target
    if kind == "chat":
        route = "/v1/chat/completions"
        body = {"model": model, "messages": [{"role": "user", "content": chat_text}],
                "max_tokens": max_tokens, "stream": stream}
    elif kind == "text":
        route = "/v1/completions"
        body = {"model": model, "prompt": text_text, "max_tokens": max_tokens,
                "stream": stream}
    elif kind == "textlist":
        route = "/v1/completions"
        body = {"model": model, "prompt": [SMALL_TEXT, text_text],
                "max_tokens": max_tokens, "stream": stream}
    elif kind == "anthropic":
        route = "/v1/messages"
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": chat_text}]}
    else:
        route = "/v1/responses"
        body = {"model": model, "input": chat_text, "max_output_tokens": max_tokens}
    with open(os.path.join(out, "bodies", rid + ".json"), "w") as f:
        json.dump(body, f)
    rows_out.append((rid, modes, route, "1" if stream else "0", cls, str(oracle_tokens),
                     str(max_tokens), "H" if (target is not None and cls != "MAXTOK") else "S"))

with open(os.path.join(out, "rows.tsv"), "w") as f:
    for r in rows_out:
        f.write("\t".join(r) + "\n")
print("ORACLE ok context=%d rows=%d" % (ctx, len(rows_out)))
PYEOF

MODEL="$MODEL" OUT="$OUT" EXPECTED_CONTEXT="$EXPECTED_CONTEXT" "$PY" "$OUT/gen.py" > "$OUT/gen.log" 2>&1
GEN_RC=$?
tail -n 3 "$OUT/gen.log" | sed 's/^/HEADER /'
if [ "$GEN_RC" -ne 0 ]; then
  echo "SUMMARY rows=0 pass=0 fail=0 void=1 oracle_rc=$GEN_RC"
  exit 1
fi

# ---------------------------------------------------------------- verdicts
cat > "$OUT/verdict.py" <<'PYEOF'
import json, sys

# argv: cls http curl_rc elapsed over_deadline body_file base_outcome
cls, http, rc, elapsed, over_deadline, body_file, base_outcome = sys.argv[1:8]
elapsed = float(elapsed)
over_deadline = float(over_deadline)
try:
    with open(body_file, errors="replace") as f:
        body = f.read()
except OSError:
    body = ""
outcome = "TIMEOUT" if rc == "28" else http


def say(verdict, why):
    print("%s|%s" % (verdict, why))


if cls == "SMALL":
    ok = http == "200" and rc == "0" and len(body) > 0
    say("PASS" if ok else "FAIL", "short prompt must be answered 200")
elif cls == "FITS":
    refused = http.startswith("4")
    say("FAIL" if refused else "PASS", "prompt within context must not get a 4xx")
elif cls == "OVER":
    named = "40960" in body
    ok = http == "400" and rc == "0" and elapsed <= over_deadline and named
    why = "need 400 within %.0fs naming 40960; got outcome=%s named=%s" % (
        over_deadline, outcome, named)
    say("PASS" if ok else "FAIL", why)
elif cls == "MAXTOK":
    ok = http == "400" and "max_tokens exceeds server limit (32768)" in body
    say("PASS" if ok else "FAIL", "existing max_tokens arm must still answer 400")
elif cls == "KEEP":
    if base_outcome == "":
        say("OBSERVED", "no base file: this outcome is the base value")
    else:
        same = outcome == base_outcome
        say("PASS" if same else "FAIL", "must equal base outcome %s" % base_outcome)
else:
    say("FAIL", "unknown class")
PYEOF

# ---------------------------------------------------------------- server control
SERVER_PID=""
START_MSG=""

start_server() {
  mode="$1"; log="$2"
  extra=""
  if [ "$mode" = "batched" ]; then extra="--continuous-batching"; fi
  nohup vllm-mlx serve "$MODEL" --host 127.0.0.1 --port "$PORT" $extra > "$log" 2>&1 < /dev/null &
  SERVER_PID=$!
  t0=$(date +%s)
  while :; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      START_MSG="server-exited-early"; return 1
    fi
    h=$(curl -s -m 5 "$BASE_URL/health" 2>/dev/null || true)
    case "$h" in
      *'"model_loaded":true'*) echo "$h" > "$OUT/health-last.json"; START_MSG=""; return 0 ;;
    esac
    if [ $(( $(date +%s) - t0 )) -gt "$START_DEADLINE" ]; then
      START_MSG="server-start-deadline"; return 1
    fi
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

base_outcome_for() {
  mode="$1"; id="$2"
  if [ ! -f "$BASE_FILE" ]; then echo ""; return; fi
  line=$(grep -F "RESULT mode=$mode id=$id " "$BASE_FILE" | head -1)
  if [ -z "$line" ]; then echo ""; return; fi
  rc=$(echo "$line" | sed -n 's/.* curl_rc=\([0-9]*\) .*/\1/p')
  http=$(echo "$line" | sed -n 's/.* http=\([0-9]*\) .*/\1/p')
  if [ "$rc" = "28" ]; then echo "TIMEOUT"; else echo "$http"; fi
}

# ---------------------------------------------------------------- run
ROWS_RUN=0; ROWS_PASS=0; ROWS_FAIL=0; ROWS_OBS=0; ROWS_EXPECTED=0

for mode in $MODES; do
  flag="s"; [ "$mode" = "batched" ] && flag="b"
  while IFS="$(printf '\t')" read -r id modes route stream cls otok mtok size <&3; do
    case "$modes" in *"$flag"*) ;; *) continue ;; esac
    ROWS_EXPECTED=$((ROWS_EXPECTED+1))
    slog="$OUT/logs/server-$mode-$id.log"
    start_server "$mode" "$slog"
    start_rc=$?
    start_msg="$START_MSG"
    if [ "$start_rc" -ne 0 ]; then
      echo "RESULT mode=$mode id=$id route=$route stream=$stream class=$cls oracle_prompt_tokens=$otok max_tokens=$mtok http=000 curl_rc=99 elapsed=0 disconnect_log=0 verdict=FAIL why=\"$start_msg\" body=\"\""
      ROWS_RUN=$((ROWS_RUN+1)); ROWS_FAIL=$((ROWS_FAIL+1))
      stop_server
      continue
    fi
    tmo="$SMALL_TIMEOUT"; [ "$size" = "H" ] && tmo="$HEAVY_TIMEOUT"
    bodyf="$OUT/resp-$mode-$id.body"
    : > "$bodyf"
    meta=$(curl -sS -N -m "$tmo" -o "$bodyf" -w '%{http_code} %{time_total}' \
      -X POST "$BASE_URL$route" -H 'content-type: application/json' \
      --data-binary "@$OUT/bodies/$id.json" 2>"$OUT/resp-$mode-$id.err")
    crc=$?
    http=$(echo "$meta" | awk '{print $1}')
    elapsed=$(echo "$meta" | awk '{print $2}')
    [ -z "$http" ] && http="000"
    [ -z "$elapsed" ] && elapsed="$tmo"
    sleep 3
    disc=$(grep -c "CLIENT DISCONNECTED" "$slog" 2>/dev/null || true)
    stop_server
    bo=$(base_outcome_for "$mode" "$id")
    vline=$("$PY" "$OUT/verdict.py" "$cls" "$http" "$crc" "$elapsed" "$OVER_DEADLINE" "$bodyf" "$bo")
    verdict="${vline%%|*}"; why="${vline#*|}"
    snippet=$(head -c 200 "$bodyf" | tr '\n\r' '  ' | tr -d '"')
    echo "RESULT mode=$mode id=$id route=$route stream=$stream class=$cls oracle_prompt_tokens=$otok max_tokens=$mtok http=$http curl_rc=$crc elapsed=$elapsed disconnect_log=$disc verdict=$verdict why=\"$why\" body=\"$snippet\""
    ROWS_RUN=$((ROWS_RUN+1))
    case "$verdict" in
      PASS) ROWS_PASS=$((ROWS_PASS+1)) ;;
      OBSERVED) ROWS_OBS=$((ROWS_OBS+1)) ;;
      *) ROWS_FAIL=$((ROWS_FAIL+1)) ;;
    esac
  done 3< "$OUT/rows.tsv"
done

LEFT=$(pgrep -f "vllm-mlx serve" | wc -l | tr -d ' ')
echo "SUMMARY rows=$ROWS_RUN expected=$ROWS_EXPECTED pass=$ROWS_PASS observed=$ROWS_OBS fail=$ROWS_FAIL servers_left=$LEFT"
if [ "$ROWS_RUN" -ne "$ROWS_EXPECTED" ] || [ "$ROWS_RUN" -eq 0 ]; then exit 1; fi
if [ "$ROWS_FAIL" -ne 0 ]; then exit 1; fi
exit 0
