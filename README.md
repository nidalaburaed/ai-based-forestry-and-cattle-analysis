# Forestry and Cattle Analysis Pipeline v3.2

**State-of-the-art AI-powered analysis system for forestry and cattle management**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project provides comprehensive analysis pipelines that process video data using cutting-edge computer vision and deep learning models.

### 🌲 Forest Analysis
Processes forest video footage to extract detailed inventory data including tree detection, species classification, biomass estimation, and comprehensive forestry planning.

### 🐄 Cattle Monitoring
Analyzes cattle video to provide herd management insights including cattle detection, behavior recognition, health monitoring, and management recommendations.

### Key Features

✅ **Multi-Model AI Pipeline**
- YOLOv8/v11 for fast tree detection
- SAM2 (Segment Anything Model 2) for precise segmentation
- DeepForest for specialized tree detection validation
- Vision Transformers for species classification

✅ **Comprehensive Forest Metrics**
- Tree detection and counting
- Species identification
- Canopy coverage analysis
- Vegetation density mapping
- Biomass and timber volume estimation
- Economic value assessment

✅ **Structured Output**
- **Sites:** Nutrient levels, timber production potential
- **Stands:** Species composition, age, size, development class
- **Management Activities:** Recommended actions with schedules
- **Economic Information:** Timber value and sales potential
- **Maps:** Thematic visualizations and plot-level data

✅ **Professional Report Generation** 🆕
- **PDF Reports:** Formal forestry plan documents
- **Excel Reports:** Multi-sheet workbooks for data analysis
- **HTML Reports:** Interactive web-based reports
- **CSV Exports:** GIS-compatible data files

✅ **Flexible Configuration**
- Multiple analysis modes (Fast, Standard, Detailed, Research)
- Configurable model selection
- GPU acceleration support
- Debug visualizations

## Quick Start

### Installation

```bash
# Clone repository
git clone
cd 

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Basic Usage

#### Option 1: Web UI (Recommended)

```bash
# Start the web application
python webapp.py

# Open browser to http://localhost:5000
# Configure all options through the intuitive UI
# - Select analysis mode (Fast/Standard/Detailed/Research)
# - Enable AI models (SAM2, DeepForest, ViT)
# - Configure frame processing
# - Generate professional reports
```

#### Option 2: Simple CLI

```bash
# Run with default settings
python analyze.py test_video/tree_test_video.mp4

# For advanced configuration, use the web UI
```

### Output

The pipeline generates:
- **ForestPlan JSON:** Complete forest inventory data
- **Frame Analysis:** Per-frame statistics and metrics
- **Professional Reports:** PDF, Excel, HTML, CSV formats
- **Debug Frames:** Annotated visualizations (if enabled)

Example output structure:
```json
{
  "id": "uuid",
  "name": "Forest Analysis - tree_test_video",
  "sites": [...],
  "stands": [...],
  "management_activities": [...],
  "economic_info": {
    "estimated_value_eur": 15000.0,
    "timber_sales_potential_m3": 250.0
  }
}
```

## Models & Methods

### Primary Detection: YOLO
- **YOLOv8/v11:** Real-time tree detection
- **Accuracy:** High precision for tree counting
- **Speed:** Fast inference (30+ FPS on GPU)

### Segmentation: SAM2
- **Segment Anything Model 2:** Zero-shot segmentation
- **Accuracy:** Precise tree crown boundaries
- **Use Case:** Detailed crown delineation

### Validation: DeepForest
- **Specialized:** Pre-trained on forest datasets
- **Accuracy:** High precision for tree detection
- **Use Case:** Cross-validation and verification

### Classification: Vision Transformers
- **ViT:** Fine-grained species identification
- **Accuracy:** State-of-the-art classification
- **Use Case:** Species-level inventory

## Documentation

- **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Comprehensive usage instructions

## Project Structure

```
forestry_and_cattle_analysis/
├── src/                           # Source code
│   ├── __init__.py               # Package initialization
│   ├── pipeline.py               # Enhanced pipeline (v3.0)
│   ├── data_structures.py        # ForestPlan data structures
│   └── video_reader.py           # Video processing utilities
├── docs/                          # Documentation
│   ├── USAGE_GUIDE.md
├── examples/                      # Usage examples
│   └── example_usage.py
├── models/                        # Model files (download separately)
│   └── .gitkeep
├── test_video/                    # Sample videos
│   └── tree_test_video.mp4
├── templates/                     # Web UI templates
│   ├── index.html
│   └── result.html
├── output/                        # Analysis results (generated)
├── analyze.py                     # CLI entry point
├── webapp.py                      # Flask web interface
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- CPU: Modern multi-core processor

### Recommended
- Python 3.10+
- 16GB+ RAM
- GPU: NVIDIA RTX 3060+ (8GB+ VRAM)
- CUDA 11.8+

### Dependencies
- OpenCV
- PyTorch
- Ultralytics (YOLO)
- DeepForest (optional)
- SAM2 (optional)
- Timm (ViT, optional)

See [requirements.txt](requirements.txt) for complete list.

## Advanced Usage

### Python API

```python
from src import (
    EnhancedForestAnalysisPipeline,
    PipelineConfig,
    AnalysisMode
)

# Configure pipeline
config = PipelineConfig(
    mode=AnalysisMode.DETAILED,
    frame_interval=2,
    use_sam2=True,
    use_deepforest=True,
    output_dir="my_analysis"
)

# Run analysis
pipeline = EnhancedForestAnalysisPipeline(config)
forest_plan = pipeline.process_video("forest_video.mp4")

# Access results
print(f"Detected {len(forest_plan.stands)} stands")
print(f"Economic value: €{forest_plan.economic_info.estimated_value_eur}")
```

See [examples/example_usage.py](examples/example_usage.py) for more examples.

### Web Interface

```bash
# Start Flask web app
python webapp.py

# Open browser to http://localhost:5000
```

## Use Cases

### Forest Inventory
- Automated tree counting and mapping
- Species composition analysis
- Stand structure assessment

### Timber Assessment
- Volume estimation
- Economic value calculation
- Harvest planning

### Conservation
- Biodiversity monitoring
- Protected area mapping
- Habitat assessment

### Research
- Forest dynamics studies
- Climate change monitoring
- Ecological modeling

## Performance

| Mode | Frames/sec | Accuracy | Use Case |
|------|-----------|----------|----------|
| Fast | 10-15 | Good | Quick overview |
| Standard | 5-8 | High | General inventory |
| Detailed | 2-4 | Very High | Detailed analysis |
| Research | 1-2 | Maximum | Scientific research |

*Benchmarks on NVIDIA RTX 3080, 1080p video*

## Contributing

Contributions welcome! Areas of interest:
- Model fine-tuning for specific regions/species
- Integration with GIS systems
- Multi-temporal analysis
- LiDAR data fusion
- Mobile deployment

## Getting Started

New to the project? Start here:

1. **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Comprehensive guide
2. **[examples/example_usage.py](examples/example_usage.py)** - Code examples

## Acknowledgments

- **YOLOv8/v11:** Ultralytics
- **SAM2:** Meta AI Research
- **DeepForest:** Weecology Lab
- **Vision Transformers:** Hugging Face, Ross Wightman (timm)

## Contact

For questions, issues, or collaboration:
- GitHub Issues: [Create an issue]
- Email: your.email@example.com

---

**Note:** This repository is for educational purposes only (to demonstrate the latest innovation and development wave in AI and IT markets, the latest technical insights) and does not include fully implemented functionalities. For the production (commercial) version, contact nidalaburaed@hotmail.com
