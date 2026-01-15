"""
Example Usage of Enhanced Forest Analysis Pipeline
===================================================

This script demonstrates various ways to use the pipeline.
"""

from src import (
    EnhancedForestAnalysisPipeline,
    PipelineConfig,
    AnalysisMode,
    ModelBackend
)
from pathlib import Path
import json


def example_basic():
    """Basic usage with default settings."""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    config = PipelineConfig(
        mode=AnalysisMode.STANDARD,
        output_dir="output/example_basic"
    )
    
    pipeline = EnhancedForestAnalysisPipeline(config)
    forest_plan = pipeline.process_video("test_video/tree_test_video.mp4")
    
    print("\nResults:")
    print(json.dumps(forest_plan.summary(), indent=2))


def example_detailed():
    """Detailed analysis with all models."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Detailed Analysis with All Models")
    print("="*80)
    
    config = PipelineConfig(
        mode=AnalysisMode.DETAILED,
        frame_interval=1,
        max_frames=50,
        use_sam2=True,
        use_deepforest=True,
        use_vit=False,  # Requires fine-tuned model
        save_debug_frames=True,
        output_dir="output/example_detailed"
    )
    
    pipeline = EnhancedForestAnalysisPipeline(config)
    forest_plan = pipeline.process_video("test_video/tree_test_video.mp4")
    
    print("\nForest Plan Summary:")
    print(f"  Sites: {len(forest_plan.sites)}")
    print(f"  Stands: {len(forest_plan.stands)}")
    print(f"  Management Activities: {len(forest_plan.management_activities)}")
    
    if forest_plan.economic_info:
        print(f"\nEconomic Information:")
        print(f"  Estimated Value: €{forest_plan.economic_info.estimated_value_eur:,.2f}")
        print(f"  Timber Potential: {forest_plan.economic_info.timber_sales_potential_m3:.1f} m³")


def example_fast():
    """Fast analysis for quick overview."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Fast Analysis")
    print("="*80)
    
    config = PipelineConfig(
        mode=AnalysisMode.FAST,
        frame_interval=5,
        max_frames=20,
        yolo_model="yolov8n.pt",  # Smallest/fastest model
        use_sam2=False,
        use_deepforest=False,
        output_dir="output/example_fast"
    )
    
    pipeline = EnhancedForestAnalysisPipeline(config)
    forest_plan = pipeline.process_video("test_video/tree_test_video.mp4")
    
    print(f"\nQuick Stats:")
    print(f"  Total Sites: {len(forest_plan.sites)}")
    print(f"  Total Stands: {len(forest_plan.stands)}")


def example_custom_config():
    """Custom configuration for specific needs."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Configuration")
    print("="*80)
    
    config = PipelineConfig(
        mode=AnalysisMode.STANDARD,
        frame_interval=2,
        max_frames=100,
        min_confidence=0.3,
        nms_threshold=0.5,
        yolo_model="yolov8l.pt",
        use_deepforest=True,
        deepforest_score_thresh=0.25,
        save_debug_frames=True,
        save_intermediate_results=True,
        output_dir="output/example_custom",
        # Custom timber prices
        timber_price_per_m3={
            "pine": 60.0,
            "spruce": 65.0,
            "birch": 50.0,
            "oak": 90.0,
            "default": 55.0
        }
    )
    
    pipeline = EnhancedForestAnalysisPipeline(config)
    forest_plan = pipeline.process_video("test_video/tree_test_video.mp4")
    
    # Access detailed results
    print("\nDetailed Results:")
    for i, site in enumerate(forest_plan.sites, 1):
        print(f"\nSite {i}:")
        print(f"  Type: {site.type}")
        print(f"  Nutrient Level: {site.nutrient_level}")
        print(f"  Timber Potential: {site.timber_potential}")
    
    for i, stand in enumerate(forest_plan.stands, 1):
        print(f"\nStand {i}:")
        print(f"  Species: {', '.join(stand.species)}")
        print(f"  Age: {stand.age} years" if stand.age else "  Age: Unknown")
        print(f"  Size: {stand.size_ha} ha" if stand.size_ha else "  Size: Unknown")
        print(f"  Development: {stand.development_class}")


def example_export_results():
    """Export results in various formats."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Export Results")
    print("="*80)
    
    config = PipelineConfig(
        mode=AnalysisMode.STANDARD,
        output_dir="output/example_export"
    )
    
    pipeline = EnhancedForestAnalysisPipeline(config)
    forest_plan = pipeline.process_video("test_video/tree_test_video.mp4")
    
    # Export to JSON
    output_dir = Path("output/example_export")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "forest_plan.json", 'w') as f:
        f.write(forest_plan.to_json(indent=2))
    print(f"✓ Exported to: {output_dir / 'forest_plan.json'}")
    
    # Export summary
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(forest_plan.summary(), f, indent=2)
    print(f"✓ Exported summary to: {output_dir / 'summary.json'}")
    
    # Export to CSV (stands)
    import csv
    with open(output_dir / "stands.csv", 'w', newline='') as f:
        if forest_plan.stands:
            writer = csv.DictWriter(f, fieldnames=['species', 'age', 'size_ha', 'development_class'])
            writer.writeheader()
            for stand in forest_plan.stands:
                writer.writerow({
                    'species': ', '.join(stand.species),
                    'age': stand.age,
                    'size_ha': stand.size_ha,
                    'development_class': stand.development_class
                })
    print(f"✓ Exported stands to: {output_dir / 'stands.csv'}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ENHANCED FOREST ANALYSIS PIPELINE - EXAMPLES")
    print("="*80)
    
    # Run examples
    # Note: Comment out examples that require models you haven't installed
    
    try:
        example_basic()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    # Uncomment to run other examples:
    # example_detailed()
    # example_fast()
    # example_custom_config()
    # example_export_results()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)

