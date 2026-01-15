"""
Example: Generate Professional Forestry Reports
================================================

This example demonstrates how to generate professional forestry reports
in multiple formats (PDF, Excel, HTML, CSV) after analyzing a video.

The reports include:
- Executive summary with aggregated statistics
- Sites, stands, and management activities
- Economic information
- Species distribution
- Detailed data tables

Author: Adrian
Date: 2024-12-15
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    EnhancedForestAnalysisPipeline,
    PipelineConfig,
    AnalysisMode,
    ReportGenerator
)


def main():
    """Generate reports from forest analysis."""
    
    # Configuration
    video_path = "test_video/tree_test_video.mp4"
    output_dir = "output"
    
    print("=" * 60)
    print("Forest Analysis Report Generation Example")
    print("=" * 60)
    
    # Step 1: Configure pipeline
    print("\n1. Configuring pipeline...")
    config = PipelineConfig(
        analysis_mode=AnalysisMode.STANDARD,
        output_dir=output_dir,
        frame_skip=5,  # Analyze every 5th frame for faster processing
        enable_segmentation=True,
        enable_species_classification=True,
        enable_height_estimation=True
    )
    
    # Step 2: Initialize pipeline
    print("2. Initializing pipeline and loading models...")
    pipeline = EnhancedForestAnalysisPipeline(config)
    
    # Step 3: Process video
    print(f"3. Processing video: {video_path}")
    print("   This may take several minutes depending on video length...")
    
    forest_plan = pipeline.process_video(video_path)
    
    # Check if aggregated stats are available
    if not pipeline.aggregated_stats:
        print("\n❌ ERROR: Aggregated statistics not available!")
        print("   Make sure you're using the updated pipeline version.")
        return
    
    print(f"\n✅ Analysis complete!")
    print(f"   - Total frames analyzed: {pipeline.aggregated_stats.get('total_frames_analyzed', 0)}")
    print(f"   - Total trees detected: {pipeline.aggregated_stats.get('total_trees_detected', 0)}")
    print(f"   - Avg trees per frame: {pipeline.aggregated_stats.get('avg_trees_per_frame', 0):.2f}")
    print(f"   - Canopy coverage: {pipeline.aggregated_stats.get('avg_canopy_coverage_percent', 0):.2f}%")
    
    # Step 4: Generate reports
    print("\n4. Generating professional reports...")
    report_gen = ReportGenerator(output_dir=output_dir)
    
    try:
        report_paths = report_gen.generate_all_reports(
            forest_plan=forest_plan,
            aggregated_stats=pipeline.aggregated_stats,
            video_path=video_path
        )
        
        print("\n✅ Reports generated successfully!")
        print("\nGenerated files:")
        for format_type, path in report_paths.items():
            print(f"   - {format_type.upper()}: {path}")
        
        # Provide recommendations
        print("\n" + "=" * 60)
        print("Report Format Guide:")
        print("=" * 60)
        print("📄 PDF Report:")
        print("   - Professional forestry plan document")
        print("   - Suitable for printing and official submissions")
        print("   - Includes all sections with formatted tables")
        
        print("\n📊 Excel Report:")
        print("   - Multi-sheet workbook with detailed data")
        print("   - Easy to analyze and manipulate data")
        print("   - Suitable for further calculations and GIS integration")
        
        print("\n🌐 HTML Report:")
        print("   - Interactive web-based report")
        print("   - Can be viewed in any browser")
        print("   - Includes responsive design for mobile devices")
        
        print("\n📋 CSV Exports:")
        print("   - Separate CSV files for sites, stands, activities, statistics")
        print("   - Compatible with GIS software (QGIS, ArcGIS)")
        print("   - Easy to import into databases and spreadsheets")
        
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("1. Review the HTML report in your browser")
        print("2. Share the PDF report with stakeholders")
        print("3. Import CSV files into GIS software for mapping")
        print("4. Use Excel report for detailed data analysis")
        
        # Check for missing optional dependencies
        if 'pdf' not in report_paths:
            print("\n⚠️  PDF generation skipped (install 'reportlab' package)")
        if 'excel' not in report_paths:
            print("⚠️  Excel generation skipped (install 'openpyxl' package)")
        
        if 'pdf' not in report_paths or 'excel' not in report_paths:
            print("\nTo enable all formats, run:")
            print("   pip install reportlab openpyxl")
        
    except Exception as e:
        print(f"\n❌ Error generating reports: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

