"""Command-line interface for the NFHS-4 district stunting pipeline."""

import argparse
import sys
from pathlib import Path

# Add src to sys.path so package can be imported directly
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nfhs.pipeline import (
    clean_stage,
    engineer_stage,
    report_stage,
    run_all_pipeline,
    segment_stage,
    train_stage,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NFHS-4 District Child Stunting Analysis & Segmentation Pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline subcommands")

    subparsers.add_parser("clean", help="Clean raw survey Excel data")
    subparsers.add_parser("engineer", help="Build composite scores and feature sets")
    subparsers.add_parser("train", help="Train regression models on training split")
    subparsers.add_parser("segment", help="Perform KMeans clustering & risk tiering")
    subparsers.add_parser("report", help="Generate 18 figures and priority reports")
    subparsers.add_parser("run-all", help="Execute entire pipeline end-to-end")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    if args.command == "clean":
        clean_stage(project_root)
    elif args.command == "engineer":
        engineer_stage(None, project_root)
    elif args.command == "train":
        train_stage(None, project_root)
    elif args.command == "segment":
        df_clean = clean_stage(project_root)
        df_feat = engineer_stage(df_clean, project_root)
        _, _, df_pred = train_stage(df_feat, project_root)
        segment_stage(df_pred, project_root)
    elif args.command == "report":
        run_all_pipeline(project_root)
    elif args.command == "run-all":
        run_all_pipeline(project_root)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
