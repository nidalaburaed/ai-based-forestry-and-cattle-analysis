# Enhanced Forest Analysis Pipeline - Usage Guide

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_enhanced.txt

# Download YOLO model (automatic on first run, or manual):
# The model will be downloaded automatically when you run the pipeline
```

### 2. Basic Usage

```bash
# Run with default settings (standard mode, YOLO only)
python pipeline_enhanced.py test_video/tree_test_video.mp4

# Run with all models enabled
python pipeline_enhanced.py test_video/tree_test_video.mp4 \
    --use-sam2 \
    --use-deepforest \
    --use-vit \
    --mode detailed

# Fast analysis (fewer frames, basic detection)
python pipeline_enhanced.py test_video/tree_test_video.mp4 --mode fast

# Save debug frames with visualizations
python pipeline_enhanced.py test_video/tree_test_video.mp4 --debug
```

## Analysis Modes

### Fast Mode
- **Purpose:** Quick overview
- **Frame Interval:** 5 seconds
- **Models:** YOLO only
- **Use Case:** Initial assessment, large videos

### Standard Mode (Default)
- **Purpose:** Balanced accuracy and speed
- **Frame Interval:** 3 seconds
- **Models:** YOLO + optional DeepForest
- **Use Case:** General forest inventory

### Detailed Mode
- **Purpose:** High accuracy analysis
- **Frame Interval:** 1 second
- **Models:** YOLO + SAM2 + DeepForest
- **Use Case:** Detailed inventory, research

### Research Mode
- **Purpose:** Maximum detail with all metrics
- **Frame Interval:** 0.5 seconds
- **Models:** All models enabled
- **Use Case:** Scientific research, validation

## Model Configuration

### YOLO (Primary Detection)

```bash
# Use different YOLO variants
python pipeline_enhanced.py video.mp4 --yolo-model yolov8n.pt  # Fastest
python pipeline_enhanced.py video.mp4 --yolo-model yolov8s.pt  # Small
python pipeline_enhanced.py video.mp4 --yolo-model yolov8m.pt  # Medium
python pipeline_enhanced.py video.mp4 --yolo-model yolov8l.pt  # Large (default)
python pipeline_enhanced.py video.mp4 --yolo-model yolov8x.pt  # Extra large

# Or use YOLOv11 (latest)
python pipeline_enhanced.py video.mp4 --yolo-model yolov11l.pt
```

### SAM2 (Segmentation)

SAM2 provides precise tree crown boundaries. Enable with `--use-sam2`.

**Installation:**
```bash
# Clone SAM2 repository
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# Download checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### DeepForest (Validation)

DeepForest is specialized for tree detection. Enable with `--use-deepforest`.

**Installation:**
```bash
pip install deepforest
```

### Vision Transformer (Species Classification)

ViT enables species identification. Enable with `--use-vit`.

**Note:** Requires fine-tuning on your specific tree species dataset.

## Output Files

### ForestPlan JSON
```
output/forest_plan_YYYYMMDD_HHMMSS.json
```

Contains:
- **Sites:** Nutrient levels, timber potential
- **Stands:** Species, age, size, development class
- **Management Activities:** Recommended actions with schedules
- **Economic Info:** Timber value estimates
- **Maps:** Analysis metadata

### Frame Results Summary
```
output/frame_results_YYYYMMDD_HHMMSS.json
```

Contains per-frame analysis:
- Tree count
- Canopy coverage
- Vegetation density
- Dominant species
- Average tree height

### Debug Frames (if --debug enabled)
```
output/debug_frame_0001.jpg
output/debug_frame_0002.jpg
...
```

Annotated frames showing:
- Bounding boxes
- Species labels
- Confidence scores
- Frame statistics

## Advanced Configuration

### Python API

```python
from pipeline_enhanced import (
    EnhancedForestAnalysisPipeline,
    PipelineConfig,
    AnalysisMode
)

# Create custom configuration
config = PipelineConfig(
    mode=AnalysisMode.DETAILED,
    frame_interval=2,
    max_frames=100,
    use_sam2=True,
    use_deepforest=True,
    use_vit=False,
    min_confidence=0.3,
    output_dir="my_analysis",
    save_debug_frames=True
)

# Run pipeline
pipeline = EnhancedForestAnalysisPipeline(config)
forest_plan = pipeline.process_video("my_video.mp4")

# Access results
print(forest_plan.summary())
print(f"Total sites: {len(forest_plan.sites)}")
print(f"Total stands: {len(forest_plan.stands)}")

# Save to custom location
with open("my_plan.json", 'w') as f:
    f.write(forest_plan.to_json())
```

### Custom Species Classification

To use ViT for species classification, you need to fine-tune on your dataset:

```python
import timm
import torch

# Load pre-trained ViT
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)

# Fine-tune on your tree species dataset
# ... training code ...

# Save fine-tuned model
torch.save(model.state_dict(), 'vit_tree_species.pth')

# Use in pipeline
config = PipelineConfig(
    use_vit=True,
    vit_model='vit_base_patch16_224',
    vit_num_classes=10
)
```

## Performance Optimization

### GPU Acceleration

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use GPU (automatic if available)
python pipeline_enhanced.py video.mp4 --device cuda

# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python pipeline_enhanced.py video.mp4
```

### Memory Management

For large videos or limited RAM:

```bash
# Process fewer frames
python pipeline_enhanced.py video.mp4 --max-frames 50

# Increase frame interval
python pipeline_enhanced.py video.mp4 --interval 5

# Use smaller YOLO model
python pipeline_enhanced.py video.mp4 --yolo-model yolov8n.pt
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use smaller model or reduce batch size
python pipeline_enhanced.py video.mp4 --yolo-model yolov8s.pt
```

**2. SAM2 Not Found**
```bash
# Install SAM2 from source
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e .
```

**3. DeepForest Errors**
```bash
# Reinstall DeepForest
pip uninstall deepforest
pip install deepforest --no-cache-dir
```

**4. Video Codec Issues**
```bash
# Install additional codecs
pip install opencv-python-headless
```

## Best Practices

### Video Quality
- **Resolution:** 1080p or higher
- **Frame Rate:** 30 fps minimum
- **Lighting:** Natural daylight, avoid shadows
- **Stability:** Use stabilized footage

### Analysis Strategy
1. Start with **fast mode** for overview
2. Use **standard mode** for general inventory
3. Apply **detailed mode** to specific areas of interest
4. Enable **all models** for research/validation

### Result Validation
- Compare YOLO and DeepForest detections
- Verify species classification manually
- Cross-reference with ground truth data
- Use debug frames to inspect detections

## Integration with GIS

Export ForestPlan to GIS formats:

```python
import json
import geopandas as gpd
from shapely.geometry import Point

# Load ForestPlan
with open('output/forest_plan.json') as f:
    plan = json.load(f)

# Convert valuable sites to GeoDataFrame
sites = []
for vsite in plan['valuable_sites']:
    if vsite['coordinates']:
        sites.append({
            'name': vsite['name'],
            'geometry': Point(vsite['coordinates']['lon'], 
                            vsite['coordinates']['lat'])
        })

gdf = gpd.GeoDataFrame(sites, crs='EPSG:4326')
gdf.to_file('forest_sites.geojson', driver='GeoJSON')
```

## Next Steps

1. **Fine-tune models** on your specific forest types
2. **Calibrate metrics** with ground truth measurements
3. **Integrate LiDAR** data for improved height estimation
4. **Add temporal analysis** for multi-season monitoring
5. **Deploy as web service** using the Flask webapp

For more information, see:
- [RESEARCH_MODELS_AND_METHODS.md](RESEARCH_MODELS_AND_METHODS.md) - Model details
- [tree_struct.py](tree_struct.py) - Data structure documentation
- [pipeline_enhanced.py](pipeline_enhanced.py) - Source code
