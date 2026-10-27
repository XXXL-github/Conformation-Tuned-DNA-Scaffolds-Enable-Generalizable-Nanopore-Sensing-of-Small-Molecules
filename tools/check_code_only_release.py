"""Fail if the publishable working tree contains anything outside the allowlist."""
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    ".gitignore", "requirements.txt",
    "code/contour_models.py", "code/contour_ml_common.py",
    "code/run_eight_analyte_classification.py", "code/run_estrogen_stack_classification.py",
    "code/run_real_sample_prediction.py", "code/run_scaffold_screening.py",
    "tests/test_workflows.py", "tests/test_scaffold.py", "tools/check_code_only_release.py",
}


def main() -> None:
    listing = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
    ).decode("utf-8")
    present = {path for path in listing.split("\0") if path and (ROOT / path).is_file()}
    unexpected, missing = present - ALLOWED, ALLOWED - present
    if unexpected or missing:
        raise SystemExit(f"Release allowlist mismatch. Unexpected: {sorted(unexpected)}; missing: {sorted(missing)}")
    for path in sorted(present):
        text = (ROOT / path).read_text(encoding="utf-8")
        if re.search(r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", text):
            raise SystemExit(f"Possible credential in {path}")
        if re.search(r"[A-Za-z]:[\\/]", text):
            raise SystemExit(f"Machine-specific absolute path in {path}")
    print(f"PASS: {len(present)} allowlisted source/documentation files; no experimental data or result files.")


if __name__ == "__main__":
    main()
