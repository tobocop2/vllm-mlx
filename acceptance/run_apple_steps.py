"""Run the pytest steps of CI's test-apple-silicon job from the ref under test."""

import re
import subprocess
import sys

import yaml

src = sys.argv[1]
only = sys.argv[2:]
with open(f"{src}/.github/workflows/ci.yml") as f:
    job = yaml.safe_load(f)["jobs"]["test-apple-silicon"]

failures = 0
for step in job["steps"]:
    run = step.get("run", "")
    if "pytest" not in run:
        continue
    name = step["name"]
    if only and not any(o in name for o in only):
        continue
    lines = [ln for ln in run.splitlines() if not ln.strip().startswith("#")]
    cmd = re.sub(r"\\\n", " ", "\n".join(lines)).strip()
    print(f"STEP_START name={name!r}", flush=True)
    print(cmd, flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=src).returncode
    print(f"STEP_RESULT name={name!r} rc={rc}", flush=True)
    failures += rc != 0
print(f"APPLE_STEPS_DONE failures={failures}", flush=True)
sys.exit(1 if failures else 0)
