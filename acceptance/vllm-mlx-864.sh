#!/bin/bash
# Acceptance script for waybarrios/vllm-mlx#864: a prompt longer than the model
# context must get a fast client error that names the limit.
#
# usage: bash acceptance/vllm-mlx-864.sh [simple|batched|all] [SRC]
#   simple   SimpleEngine   (`vllm-mlx serve MODEL`)
#   batched  BatchedEngine  (`vllm-mlx serve MODEL --continuous-batching`)
#   SRC      checkout under test (default: src); only used to find the log tag
#
# env: MODEL (default mlx-community/Qwen3-0.6B-4bit), DS_MODEL (default
#      mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit: config context 131072,
#      tokenizer model_max_length 16384), PORT (default 18864),
#      OUT (default acceptance-out), BASE_FILE (result of the base run, used by
#      KEEP rows), HEAVY_TIMEOUT (default 100 s), SMALL_TIMEOUT (default 60 s),
#      OVER_DEADLINE (default 10 s: the longest an OVER row may take),
#      CFG_TIMEOUT (default 300 s: a CFG200 row sends a 20000-token prompt),
#      ONLY (extended regex on row ids: run only the matching rows).
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
DS_MODEL="${DS_MODEL:-mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit}"
PORT="${PORT:-18864}"
OUT="${OUT:-acceptance-out}"
BASE_FILE="${BASE_FILE:-acceptance/vllm-mlx-864.base.txt}"
HEAVY_TIMEOUT="${HEAVY_TIMEOUT:-100}"
SMALL_TIMEOUT="${SMALL_TIMEOUT:-60}"
OVER_DEADLINE="${OVER_DEADLINE:-10}"
CFG_TIMEOUT="${CFG_TIMEOUT:-300}"
ONLY="${ONLY:-}"
START_DEADLINE="${START_DEADLINE:-420}"
EXPECTED_CONTEXT=40960
DS_EXPECTED_CONTEXT=131072
DS_EXPECTED_TOKENIZER_LIMIT=16384

case "$MODE_ARG" in
  simple) MODES="simple" ;;
  batched) MODES="batched" ;;
  all) MODES="simple batched" ;;
  *) echo "usage: $0 [simple|batched|all] [SRC]"; exit 2 ;;
esac

mkdir -p "$OUT/bodies" "$OUT/logs"
PY="${PYTHON:-python3}"
BASE_URL="http://127.0.0.1:$PORT"

echo "HEADER src=$SRC model=$MODEL ds_model=$DS_MODEL modes=\"$MODES\" heavy_timeout=${HEAVY_TIMEOUT}s small_timeout=${SMALL_TIMEOUT}s over_deadline=${OVER_DEADLINE}s"
if [ -d "$SRC/.git" ]; then
  echo "HEADER src_sha=$(git -C "$SRC" rev-parse HEAD)"
fi
echo "HEADER $("$PY" -m pip show vllm-mlx 2>&1 | grep -E '^(Name|Version):' | tr '\n' ' ')"

# ---------------------------------------------------------------- generator + oracle
cat > "$OUT/gen.py" <<'PYEOF'
import json, os, signal, sys

signal.alarm(900)
model = os.environ["MODEL"]
ds_model = os.environ["DS_MODEL"]
out = os.environ["OUT"]
expected_ctx = int(os.environ["EXPECTED_CONTEXT"])
ds_expected_ctx = int(os.environ["DS_EXPECTED_CONTEXT"])
ds_expected_tok = int(os.environ["DS_EXPECTED_TOKENIZER_LIMIT"])

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def declared_context(name):
    with open(hf_hub_download(name, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("max_position_embeddings")


ctx = declared_context(model)
if ctx != expected_ctx:
    print("ORACLE-VOID max_position_embeddings=%r expected %d" % (ctx, expected_ctx))
    sys.exit(3)
ds_ctx = declared_context(ds_model)
if ds_ctx != ds_expected_ctx:
    print("ORACLE-VOID ds max_position_embeddings=%r expected %d" % (ds_ctx, ds_expected_ctx))
    sys.exit(3)

toks = {model: AutoTokenizer.from_pretrained(model),
        ds_model: AutoTokenizer.from_pretrained(ds_model)}
ds_tok_limit = toks[ds_model].model_max_length
if ds_tok_limit != ds_expected_tok:
    print("ORACLE-VOID ds tokenizer model_max_length=%r expected %d" % (ds_tok_limit, ds_expected_tok))
    sys.exit(3)


def ids(name, text):
    return toks[name](text, add_special_tokens=False)["input_ids"]


def chat_tokens(name, n):
    msg = [{"role": "user", "content": "w " * n}]
    text = toks[name].apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    return len(ids(name, text))


def text_tokens(name, n):
    return len(ids(name, "w " * n))


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
# [, model, limit]: model defaults to MODEL and limit to its declared context.
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
    # A model whose tokenizer declares less (16384) than its config (131072): a
    # prompt above the tokenizer value and within the config must be answered 200
    # (the base answers it), and a prompt over the config is refused naming 131072.
    ("ds-chat-small-ns", "sb", "chat", False, "SMALL", None, 16, ds_model, ds_ctx),
    ("ds-chat-20000-ns", "sb", "chat", False, "CFG200", 20000, 1, ds_model, ds_ctx),
    ("ds-cmpl-20000-ns", "sb", "text", False, "CFG200", 20000, 1, ds_model, ds_ctx),
    ("ds-chat-overcfg-ns", "sb", "chat", False, "OVER", ds_ctx + 1, 1, ds_model, ds_ctx),
]

cache = {}


def n_for(name, kind, target):
    k = (name, "text" if kind in ("text", "textlist") else "chat", target)
    if k not in cache:
        count = text_tokens if k[1] == "text" else chat_tokens
        cache[k] = solve(lambda n: count(name, n), target)
    return cache[k]


rows_out = []
for row in ROWS:
    rid, modes, kind, stream, cls, target, max_tokens = row[:7]
    row_model, row_limit = (row[7], row[8]) if len(row) > 7 else (model, ctx)
    if target is None:
        chat_text, text_text, oracle_tokens = SMALL_CHAT, SMALL_TEXT, -1
    else:
        n = n_for(row_model, kind, target)
        chat_text = text_text = "w " * n
        oracle_tokens = target
    if kind == "chat":
        route = "/v1/chat/completions"
        body = {"model": row_model, "messages": [{"role": "user", "content": chat_text}],
                "max_tokens": max_tokens, "stream": stream}
    elif kind == "text":
        route = "/v1/completions"
        body = {"model": row_model, "prompt": text_text, "max_tokens": max_tokens,
                "stream": stream}
    elif kind == "textlist":
        route = "/v1/completions"
        body = {"model": row_model, "prompt": [SMALL_TEXT, text_text],
                "max_tokens": max_tokens, "stream": stream}
    elif kind == "anthropic":
        route = "/v1/messages"
        body = {"model": row_model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": chat_text}]}
    else:
        route = "/v1/responses"
        body = {"model": row_model, "input": chat_text, "max_output_tokens": max_tokens}
    with open(os.path.join(out, "bodies", rid + ".json"), "w") as f:
        json.dump(body, f)
    size = "S"
    if target is not None and cls == "CFG200":
        size = "X"
    elif target is not None and cls != "MAXTOK":
        size = "H"
    rows_out.append((rid, modes, route, "1" if stream else "0", cls, str(oracle_tokens),
                     str(max_tokens), size, row_model, str(row_limit)))

with open(os.path.join(out, "rows.tsv"), "w") as f:
    for r in rows_out:
        f.write("\t".join(r) + "\n")
print("ORACLE ok context=%d ds_context=%d ds_tokenizer_limit=%d rows=%d"
      % (ctx, ds_ctx, ds_tok_limit, len(rows_out)))
PYEOF

MODEL="$MODEL" DS_MODEL="$DS_MODEL" OUT="$OUT" EXPECTED_CONTEXT="$EXPECTED_CONTEXT" \
  DS_EXPECTED_CONTEXT="$DS_EXPECTED_CONTEXT" DS_EXPECTED_TOKENIZER_LIMIT="$DS_EXPECTED_TOKENIZER_LIMIT" \
  "$PY" "$OUT/gen.py" > "$OUT/gen.log" 2>&1
GEN_RC=$?
tail -n 3 "$OUT/gen.log" | sed 's/^/HEADER /'
if [ "$GEN_RC" -ne 0 ]; then
  echo "SUMMARY rows=0 pass=0 fail=0 void=1 oracle_rc=$GEN_RC"
  exit 1
fi

# ---------------------------------------------------------------- verdicts
cat > "$OUT/verdict.py" <<'PYEOF'
import json, sys

# argv: cls http curl_rc elapsed over_deadline body_file base_outcome limit
cls, http, rc, elapsed, over_deadline, body_file, base_outcome, limit = sys.argv[1:9]
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
    named = limit in body
    ok = http == "400" and rc == "0" and elapsed <= over_deadline and named
    why = "need 400 within %.0fs naming %s; got outcome=%s named=%s" % (
        over_deadline, limit, outcome, named)
    say("PASS" if ok else "FAIL", why)
elif cls == "CFG200":
    ok = http == "200" and rc == "0" and len(body) > 0
    say("PASS" if ok else "FAIL",
        "prompt within the config context must be answered 200, got outcome=%s" % outcome)
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
  mode="$1"; log="$2"; smodel="$3"
  extra=""
  if [ "$mode" = "batched" ]; then extra="--continuous-batching"; fi
  nohup vllm-mlx serve "$smodel" --host 127.0.0.1 --port "$PORT" $extra > "$log" 2>&1 < /dev/null &
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
  while IFS="$(printf '\t')" read -r id modes route stream cls otok mtok size rmodel rlimit <&3; do
    case "$modes" in *"$flag"*) ;; *) continue ;; esac
    if [ -n "$ONLY" ] && ! echo "$id" | grep -Eq "$ONLY"; then continue; fi
    ROWS_EXPECTED=$((ROWS_EXPECTED+1))
    slog="$OUT/logs/server-$mode-$id.log"
    start_server "$mode" "$slog" "$rmodel"
    start_rc=$?
    start_msg="$START_MSG"
    if [ "$start_rc" -ne 0 ]; then
      echo "RESULT mode=$mode id=$id route=$route stream=$stream class=$cls oracle_prompt_tokens=$otok max_tokens=$mtok model=$rmodel http=000 curl_rc=99 elapsed=0 disconnect_log=0 verdict=FAIL why=\"$start_msg\" body=\"\""
      ROWS_RUN=$((ROWS_RUN+1)); ROWS_FAIL=$((ROWS_FAIL+1))
      stop_server
      continue
    fi
    tmo="$SMALL_TIMEOUT"
    [ "$size" = "H" ] && tmo="$HEAVY_TIMEOUT"
    [ "$size" = "X" ] && tmo="$CFG_TIMEOUT"
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
    vline=$("$PY" "$OUT/verdict.py" "$cls" "$http" "$crc" "$elapsed" "$OVER_DEADLINE" "$bodyf" "$bo" "$rlimit")
    verdict="${vline%%|*}"; why="${vline#*|}"
    snippet=$(head -c 200 "$bodyf" | tr '\n\r' '  ' | tr -d '"')
    echo "RESULT mode=$mode id=$id route=$route stream=$stream class=$cls oracle_prompt_tokens=$otok max_tokens=$mtok model=$rmodel http=$http curl_rc=$crc elapsed=$elapsed disconnect_log=$disc verdict=$verdict why=\"$why\" body=\"$snippet\""
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
