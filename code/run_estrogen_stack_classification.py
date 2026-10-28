from __future__ import annotations

import argparse
from pathlib import Path

from contour_ml_common import ESTROGEN_CLASSES, evaluate_stack_oof, load_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description="NIL/EE2/MES Stack classification using 5-fold OOF validation.")
    parser.add_argument("--input", type=Path, required=True, help="External feature table; not bundled with code.")
    parser.add_argument("--sheet", default="Expanded_Data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, required=True)
    args = parser.parse_args()
    _, X, y = load_feature_table(args.input, ESTROGEN_CLASSES, args.sheet)
    print(evaluate_stack_oof(X, y, args.output_dir, args.random_state).to_string(index=False))


if __name__ == "__main__":
    main()
