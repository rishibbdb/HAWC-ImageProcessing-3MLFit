"""
CLI (TASKS.md Task 6)

Thin wrapper over HAWCAnalysisPipeline. No pipeline logic lives here --
only argument parsing and config overrides.

Usage:
    python -m cli --config config.yaml
    python -m cli --config config.yaml --procedure Alps
    python -m cli --config config.yaml --seed-only
"""

import argparse
import sys
from pathlib import Path
from core.config import ConfigManager
from core.pipeline import HAWCAnalysisPipeline
from core.fit_runner import FitResult


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the HAWC analysis pipeline")
    parser.add_argument("--config", required=True, help="Path to the pipeline config YAML")
    parser.add_argument(
        "--procedure", choices=["Drips", "Alps"], default=None,
        help="Override fitting_procedure from the config",
    )
    parser.add_argument(
        "--seed-only", action="store_true",
        help="Override coordinates.generate_seed_only to True (DRIPS detection only, no fit)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted Drips run from the furthest-completed step "
             "(skips re-seeding and any per-source extension/spectrum tests "
             "already completed), instead of starting over from scratch",
    )
    args = parser.parse_args(argv)

    config = ConfigManager(args.config)
    if args.procedure is not None:
        config.config["fitting_procedure"] = args.procedure
    if args.seed_only:
        config.config.setdefault("coordinates", {})["generate_seed_only"] = True

    pipeline = HAWCAnalysisPipeline(config, resume=args.resume)
    output = pipeline.run()

    if isinstance(output, FitResult):
        num_sources = len(output.model.sources)
        print(f"Fit complete: {num_sources} sources, -logL={output.log_like:.3f}, AIC={output.aic:.3f}")
        print(f"Final model directory: {output.step_dir}")
    elif isinstance(output, Path):
        print(f"Seed-only run complete. Model written to: {output}")
    else:
        print(output.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
