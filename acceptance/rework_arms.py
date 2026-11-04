"""Mutation arms for the vllm-mlx#869 rework, run on the ref under test.

usage: python acceptance/rework_arms.py SRC
Each arm changes exactly one production hunk, runs the new tests, restores the file
from the commit and proves the restore with git status.
"""

import re
import subprocess
import sys

src = sys.argv[1]
TESTS = [
    "tests/test_prompt_context_limit.py",
    "tests/test_prompt_token_count.py",
    "tests/test_truncation.py",
]
SRV = "vllm_mlx/server.py"
SIM = "vllm_mlx/engine/simple.py"
BAT = "vllm_mlx/engine/batched.py"
BASE = "vllm_mlx/engine/base.py"
TOK = "vllm_mlx/utils/tokenizer.py"

CHAT_SITE = """        try:
            _reject_chat_prompt_over_context(
                engine, prepared.messages, prepared.chat_kwargs
            )
        except HTTPException:
            tracker.finish(result="client_error")
            raise

"""
CMPL_SITE = """        try:
            _reject_prompts_over_context(
                engine, prompts[:1] if request.stream else prompts
            )
        except HTTPException:
            tracker.finish(result="client_error")
            raise

"""
CALL_ARGS = (
    "            tools=tools,\n"
    "            chat_template_kwargs=chat_template_kwargs,\n"
    "            enable_thinking=enable_thinking,\n"
)
SRV_CALL = (
    '        tools=chat_kwargs.get("tools"),\n'
    '        chat_template_kwargs=chat_kwargs.get("chat_template_kwargs"),\n'
    '        enable_thinking=chat_kwargs.get("enable_thinking"),\n'
)


def engine_arms(tag, path, ctx_guard, ctx_guard_dropped, count_guard, count_guard_dropped):
    return [
        (f"{tag}-context-limit-removed", path, None, None),
        (f"{tag}-context-limit-guard-dropped", path, ctx_guard, ctx_guard_dropped),
        (f"{tag}-count-guard-dropped", path, count_guard, count_guard_dropped),
        (f"{tag}-count-drops-tools", path, CALL_ARGS, CALL_ARGS.replace("tools=tools", "tools=None")),
        (
            f"{tag}-count-drops-template-kwargs",
            path,
            CALL_ARGS,
            CALL_ARGS.replace("chat_template_kwargs=chat_template_kwargs", "chat_template_kwargs=None"),
        ),
        (
            f"{tag}-count-drops-enable-thinking",
            path,
            CALL_ARGS,
            CALL_ARGS.replace("enable_thinking=enable_thinking", "enable_thinking=None"),
        ),
        (f"{tag}-count-returns-none", path, "        return count_prompt_tokens(self.tokenizer, prompt)\n", "        return None\n"),
    ]


SIM_CTX = "        if self._is_mllm or self._model is None:\n            return None\n        return declared_context_limit(self._model.model)\n"
BAT_CTX = "        if self._is_mllm:\n            return None\n        return declared_context_limit(self._model)\n"
SIM_CNT = "        if self._is_mllm or self._model is None:\n            return None\n        prompt = self._apply_chat_template("
BAT_CNT = "        if self._is_mllm or self._tokenizer is None:\n            return None\n        prompt = self._apply_chat_template("

ARMS = [
    ("chat-call-site-removed", SRV, CHAT_SITE, ""),
    ("completions-call-site-removed", SRV, CMPL_SITE, ""),
    ("completions-stream-checks-all-prompts", SRV, "prompts[:1] if request.stream else prompts", "prompts"),
    ("chat-count-none-not-skipped", SRV, "    if prompt_tokens is not None:\n        _reject_prompt_over_context(limit, prompt_tokens, \"messages\")\n", "    _reject_prompt_over_context(limit, prompt_tokens, \"messages\")\n"),
    ("chat-limit-none-not-skipped", SRV, "    limit = engine.context_limit\n    if limit is None:\n        return\n    prompt_tokens = engine.count_chat_prompt_tokens(", "    limit = engine.context_limit\n    prompt_tokens = engine.count_chat_prompt_tokens("),
    ("completions-limit-none-not-skipped", SRV, "    limit = engine.context_limit\n    if limit is None:\n        return\n    for prompt in prompts:", "    limit = engine.context_limit\n    for prompt in prompts:"),
    ("chat-drops-tools", SRV, SRV_CALL, SRV_CALL.replace('chat_kwargs.get("tools")', "None")),
    ("chat-drops-template-kwargs", SRV, SRV_CALL, SRV_CALL.replace('chat_kwargs.get("chat_template_kwargs")', "None")),
    ("chat-drops-enable-thinking", SRV, SRV_CALL, SRV_CALL.replace('chat_kwargs.get("enable_thinking")', "None")),
    ("boundary-ge", SRV, "    if prompt_tokens > limit:\n", "    if prompt_tokens >= limit:\n"),
    ("completions-param-named-messages", SRV, 'limit, count_prompt_tokens(engine.tokenizer, prompt), "prompt"', 'limit, count_prompt_tokens(engine.tokenizer, prompt), "messages"'),
    ("message-drops-limit", SRV, 'f"This model\'s maximum context length is {limit} tokens, "', 'f"This model\'s maximum context length is {prompt_tokens} tokens, "'),
    ("max-tokens-check-disabled", SRV, "    if requested_value > _max_request_tokens:\n        raise HTTPException(", "    if False:\n        raise HTTPException("),
    ("base-context-limit-removed", BASE, '    @property\n    def context_limit(self) -> int | None:\n        """Context length in tokens the model declares, or None when unknown."""\n        return None\n\n', ""),
    ("base-count-removed", BASE, '    def count_chat_prompt_tokens(\n        self,\n        messages: list[dict[str, Any]],\n        tools: list[dict] | None = None,\n        chat_template_kwargs: dict[str, Any] | None = None,\n        enable_thinking: bool | None = None,\n    ) -> int | None:\n        """Tokens in the prompt this engine renders for a chat, or None when unknown."""\n        return None\n\n', ""),
    ("tokenizer-always-add-special", TOK, "    add_special = bos_token is None or not prompt.startswith(bos_token)\n", "    add_special = True\n"),
    ("tokenizer-never-add-special", TOK, "    add_special = bos_token is None or not prompt.startswith(bos_token)\n", "    add_special = False\n"),
    ("tokenizer-bare-encode", TOK, "    return len(tokenizer.encode(prompt, add_special_tokens=add_special))\n", "    return len(tokenizer.encode(prompt))\n"),
    ("tokenizer-bos-must-be-prefix-dropped", TOK, "bos_token is None or not prompt.startswith(bos_token)", "bos_token is None or bos_token not in prompt"),
    ("tokenizer-bos-always-none", TOK, '    bos_token = getattr(tokenizer, "bos_token", None)\n', "    bos_token = None\n"),
]
SIM_CALL = (
    "        prompt = self._apply_chat_template(\n"
    "            messages,\n"
    + CALL_ARGS
    + "        )\n"
    "        return count_prompt_tokens(self.tokenizer, prompt)\n"
)
SWALLOW = (
    "        try:\n"
    "            prompt = self._apply_chat_template(\n"
    "                messages,\n"
    "                tools=tools,\n"
    "                chat_template_kwargs=chat_template_kwargs,\n"
    "                enable_thinking=enable_thinking,\n"
    "            )\n"
    "        except ValueError:\n"
    "            return None\n"
    "        return count_prompt_tokens(self.tokenizer, prompt)\n"
)
ARMS += [
    ("boundary-minus-two", SRV, "    if prompt_tokens > limit:\n", "    if prompt_tokens > limit - 2:\n"),
    ("body-drops-max-model-len", SRV, '                "max_model_len": limit,\n', ""),
    ("simple-limit-constrained-by-tokenizer", SIM, "        return declared_context_limit(self._model.model)\n", "        return min(declared_context_limit(self._model.model) or 10**9, self.tokenizer.model_max_length)\n"),
    ("batched-limit-constrained-by-tokenizer", BAT, "        return declared_context_limit(self._model)\n", "        return min(declared_context_limit(self._model) or 10**9, self.tokenizer.model_max_length)\n"),
    ("batched-limit-reads-args-directly", BAT, "        return declared_context_limit(self._model)\n", "        return declared_context_limit(self._model.args)\n"),
    ("simple-count-swallows-template-error", SIM, SIM_CALL, SWALLOW),
    ("batched-count-swallows-template-error", BAT, SIM_CALL, SWALLOW),
]
ARMS += engine_arms("simple", SIM, SIM_CTX, SIM_CTX.replace("self._is_mllm or ", ""), SIM_CNT, SIM_CNT.replace("self._is_mllm or ", ""))
ARMS += engine_arms("batched", BAT, BAT_CTX, "        return declared_context_limit(self._model)\n", BAT_CNT, BAT_CNT.replace("self._is_mllm or ", ""))
# unloaded guards: drop only the loaded half of each guard
ARMS += [
    ("simple-context-unloaded-guard-dropped", SIM, SIM_CTX, SIM_CTX.replace(" or self._model is None", "")),
    ("simple-count-unloaded-guard-dropped", SIM, SIM_CNT, SIM_CNT.replace(" or self._model is None", "")),
    ("batched-count-unloaded-guard-dropped", BAT, BAT_CNT, BAT_CNT.replace(" or self._tokenizer is None", "")),
]


def sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=src, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def blob(path):
    return sh(f"git hash-object {path}").stdout.strip()[:7]


def run_tests():
    r = sh("python -m pytest " + " ".join(TESTS) + " -q --tb=no -p no:cacheprovider -rA")
    out = r.stdout + r.stderr
    failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)))
    passed = sorted(set(re.findall(r"^PASSED (\S+)", out, re.M)))
    summ = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error)", ln)]
    return failed, passed, (summ[-1] if summ else out[-300:])


failed, passed, summ = run_tests()
ALL = sorted(set(passed) | set(failed))
print("BASELINE failed=%d passed=%d :: %s" % (len(failed), len(passed), summ), flush=True)
if failed or not passed:
    print("REWORK-ARMS-VOID baseline not green")
    sys.exit(1)

applied = 0
red = 0
covered = set()
for name, path, old, new in ARMS:
    text = open(f"{src}/{path}").read()
    if old is None:  # remove the whole engine property block
        m = re.search(r"    @property\n    def context_limit\(self\) -> int \| None:\n.*?\n\n(?=    def |    @)", text, re.S)
        if not m:
            print("NOT-APPLIED %s no property block" % name)
            continue
        mutated = text.replace(m.group(0), "")
    else:
        if text.count(old) != 1:
            print("NOT-APPLIED %s old_count=%d" % (name, text.count(old)))
            continue
        mutated = text.replace(old, new)
    before = blob(path)
    open(f"{src}/{path}", "w").write(mutated)
    after = blob(path)
    assert before != after, name
    changed = len(sh("git status --porcelain").stdout.splitlines())
    applied += 1
    try:
        failed, _, summ = run_tests()
    finally:
        sh(f"git checkout -- {path}")
    covered |= set(failed)
    red += bool(failed)
    print("ARM %s blob %s->%s changed_files=%s red=%d :: %s :: %s" % (
        name, before, after, changed, len(failed), summ, ", ".join(failed)), flush=True)
status = sh("git status --porcelain").stdout.strip()
print("ARM-RESTORED status=%r" % status)
never = [i for i in ALL if i not in covered]
print("NEVER-RED count=%d of %d: %s" % (len(never), len(ALL), never))
print("REWORK-ARMS-DONE applied=%d red=%d green=%d" % (applied, red, applied - red))
