#!/usr/bin/env python3
"""
Forest Analysis CLI
===================

Simple command-line interface for the Enhanced Forest Analysis Pipeline.
All configuration is now done through the web UI.

Usage:
    python analyze.py video.mp4
"""

import json
import sys
from pathlib import Path

from src import (
    EnhancedForestAnalysisPipeline,
    PipelineConfig
)


def main():
    """Main entry point for the CLI."""

    # Simple usage: just provide video path
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <video_path>")
        print("\nFor advanced configuration, use the web UI:")
        print("  python webapp.py")
        print("  Then open http://localhost:5000 in your browser")
        return 1

    video_path = Path(sys.argv[1])

    # Validate video file
    if not video_path.exists():
        print(f"Error: Video file not found: {sys.argv[1]}")
        return 1

    # Use default configuration
    config = PipelineConfig()

    # Print configuration
    print("="*80)
    print("ENHANCED FOREST ANALYSIS PIPELINE v3.1")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Mode: {config.mode.value}")
    print(f"Output: {config.output_dir}")
    print("\nNote: For advanced configuration, use the web UI (python webapp.py)")
    print("="*80)

    try:
        # Run pipeline
        pipeline = EnhancedForestAnalysisPipeline(config)
        forest_plan = pipeline.process_video(str(video_path))

        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(json.dumps(forest_plan.summary(), indent=2))
        print("="*80)
        print(f"\nResults saved to: {config.output_dir}/")

        return 0

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

