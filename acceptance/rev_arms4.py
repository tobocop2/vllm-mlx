"""Reviewer instrument for vllm-mlx#869: mutation arms and the flake run, on the ref under test.

usage: python acceptance/rev_arms.py arms|flake SRC
"""

import json
import re
import subprocess
import sys

import yaml

mode, src = sys.argv[1], sys.argv[2]
TESTS = ["tests/test_prompt_context_limit.py", "tests/test_truncation.py"]
SRVF = "vllm_mlx/server.py"
TRF = "vllm_mlx/utils/truncation.py"
SIMF = "vllm_mlx/engine/simple.py"
BATF = "vllm_mlx/engine/batched.py"
BASEF = "vllm_mlx/engine/base.py"


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
    ('positive-int-accepts-bool', TRF, 'isinstance(value, int) and not isinstance(value, bool) and value > 0', 'isinstance(value, int) and value > 0'),
    ('positive-int-accepts-zero', TRF, 'not isinstance(value, bool) and value > 0', 'not isinstance(value, bool) and value >= 0'),
    ('positive-int-accepts-float', TRF, 'isinstance(value, int) and not isinstance(value, bool)', 'isinstance(value, (int, float)) and not isinstance(value, bool)'),
    ('config-get-ignores-dict', TRF, '    if isinstance(config, dict):\n        return config.get(key)\n', ''),
    ('config-wrong-key', TRF, '_positive_int(_config_get(config, "max_position_embeddings"))', '_positive_int(_config_get(config, "max_sequence_length"))'),
    ('sentinel-not-ignored', TRF, 'if tok_val is not None and tok_val < sentinel_threshold:', 'if tok_val is not None:'),
    ('model-without-config-returns-zero', TRF, '    return resolve_context_limit(None, tokenizer)\n\n\ndef resolve_max_length', '    return 0\n\n\ndef resolve_max_length'),
    ('model-config-attrs-args-dropped', TRF, '("config", "args", "model_config")', '("config", "model_config")'),
    ('batched-property-returns-none', BATF, 'return resolve_model_context_limit(self._model, self.tokenizer)', 'return None'),
    ('simple-property-returns-none', SIMF, 'return resolve_model_context_limit(self._model.model, self.tokenizer)', 'return None'),
    ('boundary-minus-two', SRVF, 'if prompt_tokens > limit:', 'if prompt_tokens > limit - 2:'),
    ('max-tokens-check-disabled', SRVF, '    if requested_value > _max_request_tokens:\n        raise HTTPException(', '    if False:\n        raise HTTPException('),
    ('chat-renderer-guard-and-except-narrowed', SRVF, '    if limit is None or apply_template is None:\n        return\n    try:\n        prompt = apply_template(\n            messages,\n            tools=chat_kwargs.get("tools"),\n            chat_template_kwargs=chat_kwargs.get("chat_template_kwargs"),\n            enable_thinking=chat_kwargs.get("enable_thinking"),\n        )\n        prompt_tokens = _count_prompt_tokens(engine.tokenizer, prompt)\n    except Exception:\n', '    if limit is None:\n        return\n    try:\n        prompt = apply_template(\n            messages,\n            tools=chat_kwargs.get("tools"),\n            chat_template_kwargs=chat_kwargs.get("chat_template_kwargs"),\n            enable_thinking=chat_kwargs.get("enable_thinking"),\n        )\n        prompt_tokens = _count_prompt_tokens(engine.tokenizer, prompt)\n    except ValueError:\n'),
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
