"""Second reviewer instrument for vllm-mlx#869: arms the first one lacked, and wide-suite arms.

usage: python acceptance/rev_arms2.py SRC
"""

import re
import subprocess
import sys

import yaml

src = sys.argv[1]
SRVF = "vllm_mlx/server.py"
CIF = ".github/workflows/ci.yml"
TESTS = ["tests/test_prompt_context_limit.py", "tests/test_truncation.py"]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, cwd=src, capture_output=True, text=True, **kw)


def blob(path):
    return sh(f"git hash-object {path}").stdout.strip()


def head_blob(path):
    return sh(f"git rev-parse HEAD:{path}").stdout.strip()


def apple_cmd():
    with open(f"{src}/{CIF}") as f:
        job = yaml.safe_load(f)["jobs"]["test-apple-silicon"]
    cmd = None
    for step in job["steps"]:
        if step.get("name", "").startswith("Run MLX-dependent tests"):
            lines = [ln for ln in step["run"].splitlines() if not ln.strip().startswith("#")]
            cmd = re.sub(r"\\\n", " ", "\n".join(lines)).strip()
    return cmd


def parse(out):
    failed = sorted(set(re.findall(r"^FAILED (\S+)", out, re.M)))
    errs = sorted(set(re.findall(r"^ERROR (\S+)", out, re.M)))
    summ = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error)", ln)]
    return failed, errs, (summ[-1] if summ else out[-300:])


def mutate(path, old, new):
    sh(f"git checkout -- {path}")
    before = blob(path)
    assert before == head_blob(path), before
    text = open(f"{src}/{path}").read()
    n = text.count(old)
    if n != 1:
        return None, before, n
    open(f"{src}/{path}", "w").write(text.replace(old, new))
    return blob(path), before, 1


def restore(path):
    sh(f"git checkout -- {path}")
    assert blob(path) == head_blob(path)


# names of every test in the new files, from a -rA baseline
r = sh("python -m pytest " + " ".join(TESTS) + " -q --tb=no -p no:cacheprovider -rA")
names = sorted(set(re.findall(r"^PASSED (\S+)", r.stdout, re.M)))
print(f"ARM2-TESTS passed={len(names)} rc={r.returncode}", flush=True)

NARMS = [
    ("import-any-removed", SRVF, "from typing import Any\n", ""),
    ("count-uses-bare-encode", SRVF, "return len(tokenizer.encode(prompt, add_special_tokens=add_special))", "return len(tokenizer.encode(prompt))"),
    ("chat-limit-none-check-dropped", SRVF, "    if limit is None or apply_template is None:\n        return\n    try:", "    if apply_template is None:\n        return\n    try:"),
]
red_by_test = {n: [] for n in names}
for name, path, old, new in NARMS:
    after, before, n = mutate(path, old, new)
    if after is None:
        print(f"ARM2 {name} NOT-APPLIED old_count={n}", flush=True)
        continue
    changed = len(sh("git diff --name-only").stdout.split())
    r = sh("python -m pytest " + " ".join(TESTS) + " -q --tb=no -p no:cacheprovider -rfE")
    failed, errs, summ = parse(r.stdout + r.stderr)
    for t in failed + errs:
        red_by_test.setdefault(t, []).append(name)
    print(f"ARM2 {name} file={path} blob_before={before} blob_after={after} changed_files={changed} rc={r.returncode} failed={len(failed)} errors={len(errs)} summary={summ!r}", flush=True)
    for t in (failed + errs)[:4]:
        print(f"  RED {name}: {t}", flush=True)
    restore(path)

# ci.yml hunk: the new test file must be in the Apple list
before_cmd = apple_cmd()
print(f"CI-LIST present_at_head={'tests/test_prompt_context_limit.py' in before_cmd}", flush=True)
after, before, n = mutate(CIF, "            tests/test_prompt_context_limit.py \\\n", "")
if after is None:
    print(f"ARM2 ci-list-line-removed NOT-APPLIED old_count={n}", flush=True)
else:
    changed = len(sh("git diff --name-only").stdout.split())
    present = "tests/test_prompt_context_limit.py" in apple_cmd()
    print(f"ARM2 ci-list-line-removed file={CIF} blob_before={before} blob_after={after} changed_files={changed} present_after={present}", flush=True)
    print(f"  RED ci-list-line-removed: CI Apple list no longer names tests/test_prompt_context_limit.py (present_after={present})", flush=True)
restore(CIF)

# wide arms: the whole Apple MLX list, with a mutation on the server
cmd = apple_cmd()
r = sh(cmd + " -p no:cacheprovider -rfE -q --tb=no")
failed, errs, summ = parse(r.stdout + r.stderr)
print(f"WIDE-BASELINE rc={r.returncode} failed={len(failed)} errors={len(errs)} summary={summ!r}", flush=True)
WARMS = [
    ("wide-renderer-direct-attribute", SRVF, 'apply_template = getattr(engine, "_apply_chat_template", None)\n    if limit is None', 'apply_template = engine._apply_chat_template\n    if limit is None'),
]
for name, path, old, new in WARMS:
    after, before, n = mutate(path, old, new)
    if after is None:
        print(f"WIDE {name} NOT-APPLIED old_count={n}", flush=True)
        continue
    if name == "wide-both-direct":
        text = open(f"{src}/{path}").read()
        o2 = 'apply_template = getattr(engine, "_apply_chat_template", None)'
        assert text.count(o2) == 1
        open(f"{src}/{path}", "w").write(text.replace(o2, "apply_template = engine._apply_chat_template"))
        after = blob(path)
    changed = len(sh("git diff --name-only").stdout.split())
    r = sh(cmd + " -p no:cacheprovider -rfE -q --tb=no")
    failed, errs, summ = parse(r.stdout + r.stderr)
    print(f"WIDE {name} file={path} blob_before={before} blob_after={after} changed_files={changed} rc={r.returncode} failed={len(failed)} errors={len(errs)} summary={summ!r}", flush=True)
    for t in (failed + errs)[:40]:
        print(f"  WIDE-RED {name}: {t}", flush=True)
    restore(path)

never = [t for t, a in red_by_test.items() if not a]
print(f"ARM2-STATUS porcelain={sh('git status --porcelain').stdout.strip()!r}", flush=True)
print("ARMS2-DONE", flush=True)
