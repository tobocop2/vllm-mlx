"""Reviewer instrument for vllm-mlx#869: mutation arms and the flake run, on the ref under test.

usage: python acceptance/rev_arms.py arms|flake SRC
"""

import json
import re
import subprocess
import sys

import yaml

mode, src = sys.argv[1], sys.argv[2]
TESTS = ["tests/test_prompt_context_limit.py", "tests/test_prompt_token_count.py", "tests/test_truncation.py"]
SRVF = "vllm_mlx/server.py"
TRF = "vllm_mlx/utils/truncation.py"
SIMF = "vllm_mlx/engine/simple.py"
BATF = "vllm_mlx/engine/batched.py"
BASEF = "vllm_mlx/engine/base.py"
TOKF = "vllm_mlx/utils/tokenizer.py"
CIF = ".github/workflows/ci.yml"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, cwd=src, capture_output=True, text=True, **kw)


def blob(path):
    return sh(f"git hash-object {path}").stdout.strip()


def head_blob(path):
    return sh(f"git rev-parse HEAD:{path}").stdout.strip()


CHAT_BLOCK = """        try:
            _reject_chat_prompt_over_context(
                engine, prepared.messages, prepared.chat_kwargs
            )
        except HTTPException:
            tracker.finish(result="client_error")
            raise

"""
CMPL_BLOCK = """        try:
            _reject_prompts_over_context(
                engine, prompts[:1] if request.stream else prompts
            )
        except HTTPException:
            tracker.finish(result="client_error")
            raise

"""
ARMS = [
    ('chat-call-site-removed', SRVF, '        try:\n            _reject_chat_prompt_over_context(\n                engine, prepared.messages, prepared.chat_kwargs\n            )\n        except HTTPException:\n            tracker.finish(result="client_error")\n            raise\n\n', ''),
    ('completions-call-site-removed', SRVF, '        try:\n            _reject_prompts_over_context(\n                engine, prompts[:1] if request.stream else prompts\n            )\n        except HTTPException:\n            tracker.finish(result="client_error")\n            raise\n\n', ''),
    ('chat-site-tracker-finish-removed', SRVF, '        try:\n            _reject_chat_prompt_over_context(\n                engine, prepared.messages, prepared.chat_kwargs\n            )\n        except HTTPException:\n            tracker.finish(result="client_error")\n            raise\n\n', '        try:\n            _reject_chat_prompt_over_context(\n                engine, prepared.messages, prepared.chat_kwargs\n            )\n        except HTTPException:\n            raise\n\n'),
    ('completions-site-tracker-finish-removed', SRVF, '        try:\n            _reject_prompts_over_context(\n                engine, prompts[:1] if request.stream else prompts\n            )\n        except HTTPException:\n            tracker.finish(result="client_error")\n            raise\n\n', '        try:\n            _reject_prompts_over_context(\n                engine, prompts[:1] if request.stream else prompts\n            )\n        except HTTPException:\n            raise\n\n'),
    ('completions-stream-checks-all', SRVF, 'prompts[:1] if request.stream else prompts', 'prompts'),
    ('completions-checks-first-only', SRVF, 'prompts[:1] if request.stream else prompts', 'prompts[:1]'),
    ('boundary-ge', SRVF, 'if prompt_tokens > limit:', 'if prompt_tokens >= limit:'),
    ('boundary-minus-two', SRVF, 'if prompt_tokens > limit:', 'if prompt_tokens > limit - 2:'),
    ('chat-none-count-not-skipped', SRVF, '    if prompt_tokens is not None:\n        _reject_prompt_over_context(limit, prompt_tokens, "messages")', '    _reject_prompt_over_context(limit, prompt_tokens or 0, "messages")\n    if False:\n        pass'),
    ('chat-unknown-limit-not-skipped', SRVF, '    limit = engine.context_limit\n    if limit is None:\n        return\n    prompt_tokens = engine.count_chat_prompt_tokens(', '    limit = engine.context_limit\n    limit = limit or 0\n    prompt_tokens = engine.count_chat_prompt_tokens('),
    ('completions-unknown-limit-not-skipped', SRVF, '    limit = engine.context_limit\n    if limit is None:\n        return\n    for prompt in prompts:', '    limit = engine.context_limit\n    limit = limit or 0\n    for prompt in prompts:'),
    ('chat-template-error-swallowed', SRVF, '    prompt_tokens = engine.count_chat_prompt_tokens(\n        messages,\n        tools=chat_kwargs.get("tools"),\n        chat_template_kwargs=chat_kwargs.get("chat_template_kwargs"),\n        enable_thinking=chat_kwargs.get("enable_thinking"),\n    )\n', '    try:\n        prompt_tokens = engine.count_chat_prompt_tokens(\n            messages,\n            tools=chat_kwargs.get("tools"),\n            chat_template_kwargs=chat_kwargs.get("chat_template_kwargs"),\n            enable_thinking=chat_kwargs.get("enable_thinking"),\n        )\n    except Exception:\n        return\n'),
    ('chat-drops-tools', SRVF, 'tools=chat_kwargs.get("tools"),', 'tools=None,'),
    ('chat-drops-template-kwargs', SRVF, 'chat_template_kwargs=chat_kwargs.get("chat_template_kwargs"),', 'chat_template_kwargs=None,'),
    ('chat-drops-enable-thinking', SRVF, 'enable_thinking=chat_kwargs.get("enable_thinking"),', 'enable_thinking=None,'),
    ('body-param-fixed', SRVF, '"param": param,', '"param": "prompt",'),
    ('body-code', SRVF, '"code": "context_length_exceeded",', '"code": "context_length",'),
    ('body-max-model-len', SRVF, '"max_model_len": limit,', '"max_model_len": prompt_tokens,'),
    ('body-prompt-tokens', SRVF, '"prompt_tokens": prompt_tokens,', '"prompt_tokens": limit,'),
    ('body-type', SRVF, '"type": "invalid_request_error",\n                "param": param,', '"type": "server_error",\n                "param": param,'),
    ('message-drops-limit', SRVF, 'f"This model\'s maximum context length is {limit} tokens, "', '"This model\'s maximum context length is exceeded, "'),
    ('chat-param-name', SRVF, '_reject_prompt_over_context(limit, prompt_tokens, "messages")', '_reject_prompt_over_context(limit, prompt_tokens, "prompt")'),
    ('completions-param-name', SRVF, 'limit, count_prompt_tokens(engine.tokenizer, prompt), "prompt"', 'limit, count_prompt_tokens(engine.tokenizer, prompt), "messages"'),
    ('max-tokens-check-disabled', SRVF, '    if requested_value > _max_request_tokens:\n        raise HTTPException(', '    if False:\n        raise HTTPException('),
    ('server-count-always-zero', SRVF, 'limit, count_prompt_tokens(engine.tokenizer, prompt), "prompt"', 'limit, 0, "prompt"'),
    ('count-always-add-special', TOKF, 'add_special = bos_token is None or not prompt.startswith(bos_token)', 'add_special = True'),
    ('count-never-add-special', TOKF, 'add_special = bos_token is None or not prompt.startswith(bos_token)', 'add_special = False'),
    ('count-bare-encode', TOKF, 'return len(tokenizer.encode(prompt, add_special_tokens=add_special))', 'return len(tokenizer.encode(prompt))'),
    ('count-ignores-bos-attr', TOKF, 'bos_token = getattr(tokenizer, "bos_token", None)', 'bos_token = None'),
    ('declared-config-attr-only', TRF, 'for attr in ("config", "args", "model_config"):\n        limit', 'for attr in ("config",):\n        limit'),
    ('declared-attr-order-reversed', TRF, 'for attr in ("config", "args", "model_config"):\n        limit', 'for attr in ("model_config", "args", "config"):\n        limit'),
    ('declared-first-attr-wins-even-without-field', TRF, '        if limit is not None:\n            return limit\n    return None', '        return limit\n    return None'),
    ('declared-args-dropped', TRF, '("config", "args", "model_config")', '("config", "model_config")'),
    ('declared-reads-text-config', TRF, '_config_get(getattr(model, attr, None), "max_position_embeddings")', '(_config_get(getattr(model, attr, None), "max_position_embeddings") or _config_get(_config_get(getattr(model, attr, None), "text_config"), "max_position_embeddings"))'),
    ('declared-wrong-key', TRF, '_config_get(getattr(model, attr, None), "max_position_embeddings")', '_config_get(getattr(model, attr, None), "max_sequence_length")'),
    ('declared-returns-raw-not-positive', TRF, '        limit = _positive_int(\n            _config_get(getattr(model, attr, None), "max_position_embeddings")\n        )', '        limit = _config_get(getattr(model, attr, None), "max_position_embeddings")'),
    ('base-limit-default-zero', BASEF, '        """Context length in tokens the model declares, or None when unknown."""\n        return None', '        """Context length in tokens the model declares, or None when unknown."""\n        return 0'),
    ('base-count-default-zero', BASEF, '        """Tokens in the prompt this engine renders for a chat, or None when unknown."""\n        return None', '        """Tokens in the prompt this engine renders for a chat, or None when unknown."""\n        return 0'),
    ('simple-limit-mllm-guard-dropped', SIMF, '        if self._is_mllm or self._model is None:\n            return None\n        return declared_context_limit(self._model.model)', '        if self._model is None:\n            return None\n        return declared_context_limit(self._model.model)'),
    ('simple-limit-unloaded-guard-dropped', SIMF, '        if self._is_mllm or self._model is None:\n            return None\n        return declared_context_limit(self._model.model)', '        if self._is_mllm:\n            return None\n        return declared_context_limit(self._model.model)'),
    ('simple-limit-reads-wrapper', SIMF, 'declared_context_limit(self._model.model)', 'declared_context_limit(self._model)'),
    ('simple-count-mllm-guard-dropped', SIMF, '        if self._is_mllm or self._model is None:\n            return None\n        prompt = self._apply_chat_template(', '        if self._model is None:\n            return None\n        prompt = self._apply_chat_template('),
    ('simple-count-unloaded-guard-dropped', SIMF, '        if self._is_mllm or self._model is None:\n            return None\n        prompt = self._apply_chat_template(', '        if self._is_mllm:\n            return None\n        prompt = self._apply_chat_template('),
    ('simple-count-drops-tools', SIMF, '        prompt = self._apply_chat_template(\n            messages,\n            tools=tools,\n            chat_template_kwargs=chat_template_kwargs,\n            enable_thinking=enable_thinking,\n        )\n        return count_prompt_tokens(self.tokenizer, prompt)\n', '        prompt = self._apply_chat_template(\n            messages,\n            tools=None,\n            chat_template_kwargs=chat_template_kwargs,\n            enable_thinking=enable_thinking,\n        )\n        return count_prompt_tokens(self.tokenizer, prompt)\n'),
    ('simple-count-drops-enable-thinking', SIMF, '        prompt = self._apply_chat_template(\n            messages,\n            tools=tools,\n            chat_template_kwargs=chat_template_kwargs,\n            enable_thinking=enable_thinking,\n        )\n        return count_prompt_tokens(self.tokenizer, prompt)\n', '        prompt = self._apply_chat_template(\n            messages,\n            tools=tools,\n            chat_template_kwargs=chat_template_kwargs,\n            enable_thinking=None,\n        )\n        return count_prompt_tokens(self.tokenizer, prompt)\n'),
    ('simple-count-drops-template-kwargs', SIMF, '        prompt = self._apply_chat_template(\n            messages,\n            tools=tools,\n            chat_template_kwargs=chat_template_kwargs,\n            enable_thinking=enable_thinking,\n        )\n        return count_prompt_tokens(self.tokenizer, prompt)\n', '        prompt = self._apply_chat_template(\n            messages,\n            tools=tools,\n            chat_template_kwargs=None,\n            enable_thinking=enable_thinking,\n        )\n        return count_prompt_tokens(self.tokenizer, prompt)\n'),
    ('simple-count-bare-len-encode', SIMF, 'return count_prompt_tokens(self.tokenizer, prompt)', 'return len(self.tokenizer.encode(prompt))'),
    ('batched-limit-mllm-guard-dropped', BATF, '        if self._is_mllm:\n            return None\n        return declared_context_limit(self._model)', '        return declared_context_limit(self._model)'),
    ('batched-count-mllm-guard-dropped', BATF, '        if self._is_mllm or self._tokenizer is None:\n            return None\n        prompt = self._apply_chat_template(', '        if self._tokenizer is None:\n            return None\n        prompt = self._apply_chat_template('),
    ('batched-count-tokenizer-guard-dropped', BATF, '        if self._is_mllm or self._tokenizer is None:\n            return None\n        prompt = self._apply_chat_template(', '        if self._is_mllm:\n            return None\n        prompt = self._apply_chat_template('),
    ('batched-count-bare-len-encode', BATF, 'return count_prompt_tokens(self.tokenizer, prompt)', 'return len(self.tokenizer.encode(prompt))'),
]


def run_tests(extra=""):
    r = sh(
        "python -m pytest " + " ".join(TESTS) + " -q --tb=no -p no:cacheprovider -rf " + extra
    )
    out = r.stdout + r.stderr
    failed = sorted(set(re.findall(r"^FAILED (\S+)", out, re.M)))
    errs = sorted(set(re.findall(r"^ERROR (\S+)", out, re.M)))
    summ = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error)", ln)]
    return r.returncode, failed, errs, (summ[-1] if summ else out[-300:])


if mode == "arms":
    rc, failed, errs, summ = run_tests()
    print(f"ARM-BASELINE rc={rc} failed={len(failed)} errors={len(errs)} summary={summ!r}", flush=True)
    all_names = sh("python -m pytest " + " ".join(TESTS) + " -q --tb=no -p no:cacheprovider -rA").stdout
    names = sorted(set(re.findall(r"^PASSED (\S+)", all_names, re.M)))
    print(f"ARM-TESTS collected={len(names)}", flush=True)
    red_by_test = {n: [] for n in names}
    for name, path, old, new in ARMS:
        sh(f"git checkout -- {path}")
        before = blob(path)
        assert before == head_blob(path), (name, before)
        with open(f"{src}/{path}") as f:
            text = f.read()
        count = text.count(old)
        if count != 1:
            print(f"ARM {name} NOT-APPLIED old_count={count}", flush=True)
            continue
        with open(f"{src}/{path}", "w") as f:
            f.write(text.replace(old, new))
        after = blob(path)
        changed = len(sh("git diff --name-only").stdout.split())
        rc, failed, errs, summ = run_tests()
        for t in failed + errs:
            red_by_test.setdefault(t, []).append(name)
        print(
            f"ARM {name} file={path} blob_before={before} blob_after={after} changed_files={changed} "
            f"rc={rc} failed={len(failed)} errors={len(errs)} summary={summ!r}",
            flush=True,
        )
        for t in (failed + errs):
            print(f"  RED {name}: {t}", flush=True)
        sh(f"git checkout -- {path}")
        assert blob(path) == head_blob(path)
    import yaml as _y

    def _lists():
        jobs = _y.safe_load(open(f"{src}/{CIF}"))["jobs"]
        txt = []
        for jn in ("test-matrix", "test-apple-silicon"):
            txt.append(" ".join(st.get("run", "") for st in jobs[jn]["steps"]))
        return txt
    m0, a0 = _lists()
    print(f"CI-LIST at_head token_count_in_matrix={'tests/test_prompt_token_count.py' in m0} context_limit_in_apple={'tests/test_prompt_context_limit.py' in a0}", flush=True)
    for cname, cold in (("ci-token-count-test-removed", "            tests/test_prompt_token_count.py \\\n"), ("ci-context-limit-test-removed", "            tests/test_prompt_context_limit.py \\\n")):
        sh(f"git checkout -- {CIF}")
        before = blob(CIF)
        text = open(f"{src}/{CIF}").read()
        if text.count(cold) != 1:
            print(f"ARM {cname} NOT-APPLIED old_count={text.count(cold)}", flush=True)
            continue
        open(f"{src}/{CIF}", "w").write(text.replace(cold, ""))
        m1, a1 = _lists()
        print(f"ARM {cname} file={CIF} blob_before={before} blob_after={blob(CIF)} changed_files={len(sh('git diff --name-only').stdout.split())} token_count_in_matrix={'tests/test_prompt_token_count.py' in m1} context_limit_in_apple={'tests/test_prompt_context_limit.py' in a1}", flush=True)
        sh(f"git checkout -- {CIF}")
    print(f"ARM-RESTORED status={sh('git status --porcelain').stdout.strip()!r}", flush=True)
    never = [t for t, a in red_by_test.items() if not a]
    print(f"NEVER-RED count={len(never)} of {len(names)}", flush=True)
    for t in never:
        print(f"NEVER-RED {t}", flush=True)
    print("ARMS-DONE", flush=True)
elif mode == "flake":
    with open(f"{src}/.github/workflows/ci.yml") as f:
        job = yaml.safe_load(f)["jobs"]["test-apple-silicon"]
    cmd = None
    for step in job["steps"]:
        if step.get("name", "").startswith("Run MLX-dependent tests"):
            lines = [ln for ln in step["run"].splitlines() if not ln.strip().startswith("#")]
            cmd = re.sub(r"\\\n", " ", "\n".join(lines)).strip()
    assert cmd and "tests/test_prompt_context_limit.py" in cmd, cmd
    suite = subprocess.Popen(
        cmd + " -n 3 -p no:cacheprovider", shell=True, cwd=src,
        stdout=open("suite-parallel.log", "w"), stderr=subprocess.STDOUT,
    )
    print(f"FLAKE-SUITE pid={suite.pid} started", flush=True)
    fails = 0
    tests_each = set()
    for i in range(1, 21):
        r = sh("python -m pytest tests/test_prompt_context_limit.py -q --tb=short -p no:cacheprovider")
        out = r.stdout + r.stderr
        m = re.search(r"(\d+) passed", out)
        n = int(m.group(1)) if m else 0
        tests_each.add(n)
        last = out.strip().splitlines()[-1] if out.strip() else ""
        if r.returncode != 0 or n == 0:
            fails += 1
            print(out[-1500:], flush=True)
        print(f"FLAKE rep={i} rc={r.returncode} passed={n} line={last!r}", flush=True)
    suite_rc = suite.wait()
    tail = open("suite-parallel.log").read().strip().splitlines()[-1]
    print(f"FLAKE-SUITE rc={suite_rc} line={tail!r}", flush=True)
    print(f"FLAKE-DONE reps=20 failures={fails} tests_each={sorted(tests_each)} suite_rc={suite_rc}", flush=True)
