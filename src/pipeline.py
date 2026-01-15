"""
Enhanced Forest Analysis Pipeline v3.0
=======================================
State-of-the-art video-based forest inventory system using latest AI models.

Features:
- YOLOv8/v11 for tree detection
- SAM2 for precise segmentation
- DeepForest for validation
- Vision Transformers for species classification
- Biomass estimation for economic analysis
- Comprehensive ForestPlan generation

Models Used:
- Primary Detection: YOLOv8l/YOLOv11
- Segmentation: SAM2 (Segment Anything Model 2)
- Validation: DeepForest 1.5+
- Classification: Vision Transformer (ViT)
- Biomass: Deep learning regression

Author: Adrian
Date: December 2024
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Generator, Union
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
import json
import os
from pathlib import Path
import logging

from .video_reader import VideoReader
from .data_structures import (
    ForestPlan, Site, Stand, ManagementActivity,
    ValuableSite, MapLayer, EconomicInfo
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================
class AnalysisMode(Enum):
    """Analysis modes for different scenarios."""
    FAST = "fast"           # Quick analysis, fewer frames, basic models
    STANDARD = "standard"   # Balanced performance and accuracy
    DETAILED = "detailed"   # Maximum accuracy, all models, more frames
    RESEARCH = "research"   # Full analysis with all metrics and visualizations


class ModelBackend(Enum):
    """Available model backends."""
    YOLO_V8 = "yolov8"
    YOLO_V11 = "yolov11"
    DEEPFOREST = "deepforest"
    SAM2 = "sam2"
    VIT = "vit"


@dataclass
class PipelineConfig:
    """Enhanced pipeline configuration with latest models."""
    
    # Video processing
    frame_interval: int = 3                    # Seconds between frames
    max_frames: Optional[int] = None           # Limit frames (None = all)
    target_resolution: Tuple[int, int] = (1920, 1080)  # Target frame size
    
    # Analysis mode
    mode: AnalysisMode = AnalysisMode.STANDARD
    
    # Detection settings
    min_tree_height_ratio: float = 0.15        # Min tree height as % of frame
    min_confidence: float = 0.25               # Detection confidence threshold
    nms_threshold: float = 0.45                # Non-maximum suppression threshold
    
    # Color thresholds (HSV)
    green_threshold: Tuple[int, int] = (35, 85)  # Vegetation
    brown_threshold: Tuple[int, int] = (10, 25)  # Bark
    
    # Output
    output_dir: str = "output"
    save_debug_frames: bool = False
    save_intermediate_results: bool = True
    generate_visualizations: bool = True
    
    # Model selection
    primary_detector: ModelBackend = ModelBackend.YOLO_V8
    use_sam2: bool = True                      # Use SAM2 for segmentation
    use_deepforest: bool = True                # Use DeepForest for validation
    use_vit: bool = False                      # Use ViT for species classification
    
    # YOLO configuration
    yolo_model: str = "yolov8l.pt"             # yolov8n/s/m/l/x or yolov11n/s/m/l/x
    yolo_device: str = "cuda"                  # cuda, cpu, or mps (Mac)
    
    # DeepForest configuration
    deepforest_score_thresh: float = 0.3       # Confidence threshold
    deepforest_patch_size: int = 400           # Patch size for processing
    
    # SAM2 configuration
    sam2_model: str = "sam2_hiera_large.yaml"  # sam2_hiera_tiny/small/base_plus/large
    sam2_checkpoint: str = "sam2_hiera_large.pt"
    sam2_points_per_side: int = 32             # Grid points for automatic segmentation
    
    # ViT configuration
    vit_model: str = "vit_base_patch16_224"    # Vision Transformer model
    vit_num_classes: int = 10                  # Number of tree species
    
    # Biomass estimation
    estimate_biomass: bool = False             # Enable biomass estimation
    biomass_model: Optional[str] = None        # Path to biomass model
    
    # Economic parameters
    timber_price_per_m3: Dict[str, float] = field(default_factory=lambda: {
        "pine": 55.0,
        "spruce": 60.0,
        "birch": 45.0,
        "oak": 80.0,
        "default": 50.0
    })
    
    # Site classification parameters
    density_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "low": (0, 0.3),
        "medium": (0.3, 0.6),
        "high": (0.6, 1.0)
    })


# =============================================================================
# MODEL LOADERS
# =============================================================================
class ModelManager:
    """Manages loading and initialization of all AI models."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models = {}
        logger.info("Initializing ModelManager...")
        
    def load_yolo(self):
        """Load YOLO model (v8 or v11)."""
        try:
            from ultralytics import YOLO
            import torch

            # Auto-detect device if CUDA is not available
            device = self.config.yolo_device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            logger.info(f"Loading YOLO model: {self.config.yolo_model} on {device}")
            model = YOLO(self.config.yolo_model)
            model.to(device)
            self.models['yolo'] = model
            logger.info(f"✓ YOLO model loaded successfully on {device}")
            return model
        except ImportError:
            logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            return None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return None

    def load_deepforest(self):
        """Load DeepForest model."""
        try:
            from deepforest import main
            logger.info("Loading DeepForest model...")
            model = main.deepforest()
            model.use_release()
            self.models['deepforest'] = model
            logger.info("✓ DeepForest model loaded successfully")
            return model
        except ImportError:
            logger.error("deepforest package not installed. Install with: pip install deepforest")
            return None
        except Exception as e:
            logger.error(f"Failed to load DeepForest model: {e}")
            return None

    def load_sam2(self):
        """Load SAM2 (Segment Anything Model 2)."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info(f"Loading SAM2 model: {self.config.sam2_model}")
            sam2_model = build_sam2(
                self.config.sam2_model,
                self.config.sam2_checkpoint,
                device=self.config.yolo_device
            )
            predictor = SAM2ImagePredictor(sam2_model)
            self.models['sam2'] = predictor
            logger.info("✓ SAM2 model loaded successfully")
            return predictor
        except ImportError:
            logger.error("SAM2 not installed. Install from: https://github.com/facebookresearch/segment-anything-2")
            return None
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            return None

    def load_vit(self):
        """Load Vision Transformer for species classification."""
        try:
            import timm
            import torch

            logger.info(f"Loading Vision Transformer: {self.config.vit_model}")
            model = timm.create_model(
                self.config.vit_model,
                pretrained=True,
                num_classes=self.config.vit_num_classes
            )
            model.eval()
            model.to(self.config.yolo_device)
            self.models['vit'] = model
            logger.info("✓ Vision Transformer loaded successfully")
            return model
        except ImportError:
            logger.error("timm package not installed. Install with: pip install timm")
            return None
        except Exception as e:
            logger.error(f"Failed to load ViT model: {e}")
            return None

    def load_all_models(self):
        """Load all configured models."""
        logger.info("Loading all configured models...")

        # Always load primary detector
        if self.config.primary_detector in [ModelBackend.YOLO_V8, ModelBackend.YOLO_V11]:
            self.load_yolo()

        # Load optional models
        if self.config.use_deepforest:
            self.load_deepforest()

        if self.config.use_sam2:
            self.load_sam2()

        if self.config.use_vit:
            self.load_vit()

        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        return self.models


# =============================================================================
# FRAME ANALYSIS - Enhanced with Latest Models
# =============================================================================
@dataclass
class TreeDetection:
    """Represents a detected tree."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0
    class_name: str = "tree"
    species: Optional[str] = None
    crown_area: Optional[float] = None
    mask: Optional[np.ndarray] = None
    estimated_height: Optional[float] = None
    estimated_dbh: Optional[float] = None  # Diameter at breast height
    health_status: Optional[str] = None


@dataclass
class FrameAnalysisResult:
    """Results from analyzing a single frame."""
    frame_number: int
    timestamp: float
    detections: List[TreeDetection]
    total_trees: int
    canopy_coverage: float
    vegetation_density: float
    dominant_species: Optional[str] = None
    average_tree_height: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedFrameAnalyzer:
    """
    Enhanced frame analyzer using state-of-the-art models.
    Supports YOLO, DeepForest, SAM2, and ViT.
    """

    def __init__(self, config: PipelineConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.models = model_manager.models
        logger.info("EnhancedFrameAnalyzer initialized")

    def detect_trees_yolo(self, frame: np.ndarray) -> List[TreeDetection]:
        """Detect trees using YOLO."""
        if 'yolo' not in self.models:
            logger.warning("YOLO model not loaded, skipping detection")
            return []

        model = self.models['yolo']
        results = model.predict(
            frame,
            conf=self.config.min_confidence,
            iou=self.config.nms_threshold,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                detection = TreeDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=model.names[cls_id] if hasattr(model, 'names') else "tree"
                )
                detections.append(detection)

        logger.debug(f"YOLO detected {len(detections)} trees")
        return detections

    def detect_trees_deepforest(self, frame: np.ndarray) -> List[TreeDetection]:
        """Detect trees using DeepForest."""
        if 'deepforest' not in self.models:
            logger.warning("DeepForest model not loaded, skipping detection")
            return []

        model = self.models['deepforest']

        # DeepForest expects RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        predictions = model.predict_image(
            image=rgb_frame,
            return_plot=False,
            score_thresh=self.config.deepforest_score_thresh
        )

        detections = []
        if predictions is not None and len(predictions) > 0:
            for _, row in predictions.iterrows():
                detection = TreeDetection(
                    bbox=(int(row['xmin']), int(row['ymin']),
                          int(row['xmax']), int(row['ymax'])),
                    confidence=float(row['score']),
                    class_name="tree"
                )
                detections.append(detection)

        logger.debug(f"DeepForest detected {len(detections)} trees")
        return detections

    def segment_trees_sam2(self, frame: np.ndarray,
                          detections: List[TreeDetection]) -> List[TreeDetection]:
        """Refine tree detections using SAM2 segmentation."""
        if 'sam2' not in self.models or not detections:
            return detections

        predictor = self.models['sam2']

        # Convert BGR to RGB for SAM2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb_frame)

        refined_detections = []
        for detection in detections:
            # Use bbox center as prompt point
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            point_coords = np.array([[center_x, center_y]])
            point_labels = np.array([1])  # 1 = foreground

            try:
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False
                )

                if len(masks) > 0:
                    mask = masks[0]
                    detection.mask = mask
                    detection.crown_area = float(np.sum(mask))

                    # Update bbox to match mask
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0:
                        detection.bbox = (
                            int(x_indices.min()),
                            int(y_indices.min()),
                            int(x_indices.max()),
                            int(y_indices.max())
                        )

                refined_detections.append(detection)
            except Exception as e:
                logger.warning(f"SAM2 segmentation failed for detection: {e}")
                refined_detections.append(detection)

        logger.debug(f"SAM2 refined {len(refined_detections)} detections")
        return refined_detections

    def classify_species_vit(self, frame: np.ndarray,
                            detections: List[TreeDetection]) -> List[TreeDetection]:
        """Classify tree species using Vision Transformer."""
        if 'vit' not in self.models or not detections:
            return detections

        import torch
        from torchvision import transforms

        model = self.models['vit']
        device = self.config.yolo_device

        # Define transforms for ViT
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Species mapping (example - should be loaded from config)
        species_map = {
            0: "pine", 1: "spruce", 2: "birch", 3: "oak", 4: "maple",
            5: "ash", 6: "beech", 7: "fir", 8: "larch", 9: "unknown"
        }

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            tree_crop = frame[y1:y2, x1:x2]

            if tree_crop.size == 0:
                continue

            try:
                # Prepare image
                img_tensor = transform(tree_crop).unsqueeze(0).to(device)

                # Predict
                with torch.no_grad():
                    outputs = model(img_tensor)
                    _, predicted = torch.max(outputs, 1)
                    species_id = predicted.item()

                detection.species = species_map.get(species_id, "unknown")
            except Exception as e:
                logger.warning(f"Species classification failed: {e}")
                detection.species = "unknown"

        return detections

    def estimate_tree_metrics(self, detection: TreeDetection,
                             frame_height: int) -> TreeDetection:
        """Estimate tree height and DBH from detection."""
        x1, y1, x2, y2 = detection.bbox
        tree_height_pixels = y2 - y1
        tree_width_pixels = x2 - x1

        # Rough estimation (assumes camera at known distance and angle)
        # These formulas should be calibrated with ground truth data
        height_ratio = tree_height_pixels / frame_height

        if height_ratio >= self.config.min_tree_height_ratio:
            # Estimate height in meters (rough approximation)
            # Assumes average tree in frame is ~15m tall
            detection.estimated_height = height_ratio * 15.0

            # Estimate DBH using allometric relationship
            # DBH (cm) ≈ Height (m) * 2.5 (very rough approximation)
            if detection.estimated_height:
                detection.estimated_dbh = detection.estimated_height * 2.5

        return detection

    def calculate_canopy_coverage(self, frame: np.ndarray,
                                  detections: List[TreeDetection]) -> float:
        """Calculate canopy coverage percentage."""
        if not detections:
            return 0.0

        frame_area = frame.shape[0] * frame.shape[1]
        total_crown_area = 0.0

        for detection in detections:
            if detection.mask is not None:
                total_crown_area += detection.crown_area or 0.0
            else:
                # Use bbox area as approximation
                x1, y1, x2, y2 = detection.bbox
                total_crown_area += (x2 - x1) * (y2 - y1)

        coverage = min(total_crown_area / frame_area, 1.0)
        return coverage

    def calculate_vegetation_density(self, frame: np.ndarray) -> float:
        """Calculate vegetation density using HSV color analysis."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green vegetation mask
        lower_green = np.array([self.config.green_threshold[0], 40, 40])
        upper_green = np.array([self.config.green_threshold[1], 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Calculate density
        green_pixels = np.sum(green_mask > 0)
        total_pixels = frame.shape[0] * frame.shape[1]
        density = green_pixels / total_pixels

        return density

    def analyze_frame(self, frame: np.ndarray, frame_number: int,
                     timestamp: float) -> FrameAnalysisResult:
        """Perform complete analysis on a single frame."""
        logger.info(f"Analyzing frame {frame_number} at {timestamp:.2f}s")

        # Step 1: Primary detection
        if self.config.primary_detector in [ModelBackend.YOLO_V8, ModelBackend.YOLO_V11]:
            detections = self.detect_trees_yolo(frame)
        else:
            detections = []

        # Step 2: Validation with DeepForest (if enabled)
        if self.config.use_deepforest and 'deepforest' in self.models:
            df_detections = self.detect_trees_deepforest(frame)
            # Merge or validate detections (simple approach: use YOLO if available, else DeepForest)
            if not detections:
                detections = df_detections

        # Step 3: Refine with SAM2 segmentation
        if self.config.use_sam2 and detections:
            detections = self.segment_trees_sam2(frame, detections)

        # Step 4: Species classification
        if self.config.use_vit and detections:
            detections = self.classify_species_vit(frame, detections)

        # Step 5: Estimate tree metrics
        frame_height = frame.shape[0]
        for detection in detections:
            self.estimate_tree_metrics(detection, frame_height)

        # Step 6: Calculate frame-level metrics
        canopy_coverage = self.calculate_canopy_coverage(frame, detections)
        vegetation_density = self.calculate_vegetation_density(frame)

        # Determine dominant species
        species_counts = {}
        for det in detections:
            if det.species:
                species_counts[det.species] = species_counts.get(det.species, 0) + 1
        dominant_species = max(species_counts, key=species_counts.get) if species_counts else None

        # Calculate average tree height
        heights = [d.estimated_height for d in detections if d.estimated_height]
        avg_height = np.mean(heights) if heights else None

        result = FrameAnalysisResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            total_trees=len(detections),
            canopy_coverage=canopy_coverage,
            vegetation_density=vegetation_density,
            dominant_species=dominant_species,
            average_tree_height=avg_height,
            metadata={
                'species_distribution': species_counts,
                'frame_shape': frame.shape
            }
        )

        logger.info(f"Frame {frame_number}: {len(detections)} trees, "
                   f"{canopy_coverage:.2%} canopy, {vegetation_density:.2%} vegetation")

        return result


# =============================================================================
# FOREST PLAN GENERATOR
# =============================================================================
class ForestPlanGenerator:
    """Generates comprehensive ForestPlan from analysis results."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def generate_plan(self, video_path: Union[str, int],
                     frame_results: List[FrameAnalysisResult]) -> Tuple[ForestPlan, Dict[str, Any]]:
        """Generate ForestPlan from frame analysis results.

        Returns:
            Tuple of (ForestPlan, aggregated_stats_dict)
        """
        logger.info("Generating ForestPlan...")

        # Handle camera index (int) vs file path (str)
        if isinstance(video_path, int):
            plan_name = f"Forest Analysis - Camera {video_path}"
        else:
            plan_name = f"Forest Analysis - {Path(video_path).stem}"
        plan = ForestPlan(plan_name)

        # Aggregate statistics
        total_detections = sum(r.total_trees for r in frame_results)
        avg_trees_per_frame = total_detections / len(frame_results) if frame_results else 0
        avg_canopy = np.mean([r.canopy_coverage for r in frame_results])
        avg_vegetation = np.mean([r.vegetation_density for r in frame_results])

        # Collect all species
        all_species = []
        for result in frame_results:
            for det in result.detections:
                if det.species:
                    all_species.append(det.species)

        species_counts = {}
        for species in all_species:
            species_counts[species] = species_counts.get(species, 0) + 1

        # Calculate additional statistics
        all_heights = []
        all_dbh = []
        for result in frame_results:
            for det in result.detections:
                if det.estimated_height:
                    all_heights.append(det.estimated_height)
                if det.estimated_dbh:
                    all_dbh.append(det.estimated_dbh)

        avg_tree_height = float(np.mean(all_heights)) if all_heights else None
        avg_dbh = float(np.mean(all_dbh)) if all_dbh else None

        # Build aggregated statistics dictionary
        aggregated_stats = {
            'total_frames_analyzed': len(frame_results),
            'total_trees_detected': int(total_detections),
            'avg_trees_per_frame': round(float(avg_trees_per_frame), 2),
            'avg_canopy_coverage': round(float(avg_canopy), 4),
            'avg_canopy_coverage_percent': round(float(avg_canopy) * 100, 2),
            'avg_vegetation_density': round(float(avg_vegetation), 4),
            'avg_vegetation_density_percent': round(float(avg_vegetation) * 100, 2),
            'species_distribution': species_counts,
            'avg_tree_height_m': round(avg_tree_height, 2) if avg_tree_height else None,
            'avg_dbh_cm': round(avg_dbh, 2) if avg_dbh else None,
            'video_path': str(video_path),  # Convert to string for JSON serialization
            'analysis_date': str(date.today())
        }

        # 1. Generate Sites
        site = self._generate_site(avg_canopy, avg_vegetation, species_counts)
        plan.add_site(site)

        # 2. Generate Stands
        stands = self._generate_stands(frame_results, species_counts)
        for stand in stands:
            plan.add_stand(stand)

        # 3. Generate Management Activities
        activities = self._generate_management_activities(
            avg_trees_per_frame, avg_canopy, species_counts
        )
        for activity in activities:
            plan.add_management_activity(activity)

        # 4. Economic Information
        economic_info = self._generate_economic_info(frame_results, species_counts)
        plan.set_economic_info(economic_info)

        # 5. Add metadata map layer
        map_layer = MapLayer(
            name="Video Analysis Coverage",
            description=f"Analysis from {len(frame_results)} frames",
            metadata={
                'video_path': str(video_path),  # Convert to string for JSON serialization
                'total_frames_analyzed': len(frame_results),
                'analysis_date': str(date.today())
            }
        )
        plan.add_map(map_layer)

        logger.info(f"ForestPlan generated: {plan.summary()}")
        return plan, aggregated_stats

    def _generate_site(self, canopy_coverage: float, vegetation_density: float,
                      species_counts: Dict[str, int]) -> Site:
        """Generate Site based on forest characteristics."""

        # Determine site type
        if canopy_coverage > 0.7:
            site_type = "dense_forest"
        elif canopy_coverage > 0.4:
            site_type = "grove"
        else:
            site_type = "sparse_woodland"

        # Determine nutrient level (based on vegetation density)
        if vegetation_density > 0.6:
            nutrient_level = "high"
        elif vegetation_density > 0.3:
            nutrient_level = "medium"
        else:
            nutrient_level = "low"

        # Determine timber potential (based on species and density)
        valuable_species = ['pine', 'spruce', 'oak']
        has_valuable = any(s in species_counts for s in valuable_species)

        if has_valuable and canopy_coverage > 0.5:
            timber_potential = "high"
        elif canopy_coverage > 0.3:
            timber_potential = "medium"
        else:
            timber_potential = "low"

        site = Site(
            name="Primary Analysis Site",
            type=site_type,
            nutrient_level=nutrient_level,
            timber_potential=timber_potential,
            notes=f"Canopy coverage: {canopy_coverage:.1%}, "
                  f"Vegetation density: {vegetation_density:.1%}"
        )

        return site

    def _generate_stands(self, frame_results: List[FrameAnalysisResult],
                        species_counts: Dict[str, int]) -> List[Stand]:
        """Generate Stand information."""
        stands = []

        # Group by dominant species
        species_groups = {}
        for result in frame_results:
            if result.dominant_species:
                if result.dominant_species not in species_groups:
                    species_groups[result.dominant_species] = []
                species_groups[result.dominant_species].append(result)

        for species, results in species_groups.items():
            # Calculate average metrics for this species group
            avg_trees = np.mean([r.total_trees for r in results])
            heights = []
            for r in results:
                for det in r.detections:
                    if det.species == species and det.estimated_height:
                        heights.append(det.estimated_height)

            avg_height = np.mean(heights) if heights else None

            # Estimate age from height (very rough approximation)
            # Assumes ~0.3m growth per year for common species
            estimated_age = int(avg_height / 0.3) if avg_height else None

            # Determine development class
            if avg_height and avg_height > 15:
                dev_class = "mature"
            elif avg_height and avg_height > 8:
                dev_class = "pole"
            elif avg_height and avg_height > 3:
                dev_class = "sapling"
            else:
                dev_class = "seedling"

            # Estimate size in hectares (rough approximation)
            # Assumes video covers approximately 0.1 ha per frame
            size_ha = len(results) * 0.1

            stand = Stand(
                species=[species],
                age=estimated_age,
                size_ha=round(size_ha, 2),
                development_class=dev_class,
                notes=f"Average {avg_trees:.1f} trees per frame, "
                      f"average height {avg_height:.1f}m" if avg_height else None
            )
            stands.append(stand)

        return stands

    def _generate_management_activities(self, avg_trees_per_frame: float,
                                       canopy_coverage: float,
                                       species_counts: Dict[str, int]) -> List[ManagementActivity]:
        """Generate recommended management activities."""
        activities = []

        # Thinning recommendation
        if canopy_coverage > 0.7:
            activity = ManagementActivity(
                description="Selective thinning to improve forest health and growth",
                start_date=date.today().replace(year=date.today().year + 1),
                end_date=date.today().replace(year=date.today().year + 1, month=6),
                responsible_party="Forest Manager",
                notes="High canopy coverage detected. Thinning recommended to reduce competition."
            )
            activities.append(activity)

        # Harvesting recommendation for mature stands
        if avg_trees_per_frame > 20 and canopy_coverage > 0.5:
            # Estimate harvestable volume (very rough)
            # Assumes average tree volume of 0.5 m³
            estimated_volume = avg_trees_per_frame * 0.3 * 0.5  # 30% harvest rate

            activity = ManagementActivity(
                description="Commercial thinning harvest",
                start_date=date.today().replace(year=date.today().year + 2),
                harvest_estimate_m3=round(estimated_volume, 1),
                responsible_party="Harvesting Contractor",
                notes="Mature stand suitable for commercial harvest"
            )
            activities.append(activity)

        # Regeneration recommendation
        if canopy_coverage < 0.3:
            activity = ManagementActivity(
                description="Forest regeneration - planting or natural regeneration support",
                start_date=date.today().replace(year=date.today().year + 1, month=4),
                responsible_party="Silviculture Team",
                notes="Low canopy coverage. Consider regeneration activities."
            )
            activities.append(activity)

        return activities

    def _generate_economic_info(self, frame_results: List[FrameAnalysisResult],
                               species_counts: Dict[str, int]) -> EconomicInfo:
        """Generate economic value estimates."""

        # Calculate total estimated timber volume
        total_volume = 0.0
        total_value = 0.0

        for result in frame_results:
            for detection in result.detections:
                if detection.estimated_height and detection.estimated_dbh:
                    # Rough volume estimation using cylinder approximation
                    # V = π * (DBH/2)² * Height * form factor
                    dbh_m = detection.estimated_dbh / 100  # Convert cm to m
                    height_m = detection.estimated_height
                    form_factor = 0.4  # Typical for conifers

                    volume = np.pi * (dbh_m / 2) ** 2 * height_m * form_factor
                    total_volume += volume

                    # Get species-specific price
                    species = detection.species or "default"
                    price = self.config.timber_price_per_m3.get(
                        species,
                        self.config.timber_price_per_m3["default"]
                    )
                    total_value += volume * price

        # Average across frames to get per-hectare estimates
        # Assumes each frame represents ~0.1 ha
        area_ha = len(frame_results) * 0.1
        volume_per_ha = total_volume / area_ha if area_ha > 0 else 0
        value_per_ha = total_value / area_ha if area_ha > 0 else 0

        economic_info = EconomicInfo(
            estimated_value_eur=round(total_value, 2),
            timber_sales_potential_m3=round(total_volume, 2),
            notes=f"Estimated {volume_per_ha:.1f} m³/ha, "
                  f"€{value_per_ha:.0f}/ha. "
                  f"Based on analysis of {len(frame_results)} frames covering ~{area_ha:.1f} ha."
        )

        return economic_info


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class EnhancedForestAnalysisPipeline:
    """Main pipeline orchestrating the entire analysis process."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.model_manager.load_all_models()
        self.analyzer = EnhancedFrameAnalyzer(config, self.model_manager)
        self.plan_generator = ForestPlanGenerator(config)

        # Initialize aggregated statistics (will be populated after processing)
        self.aggregated_stats: Optional[Dict[str, Any]] = None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def process_video(self, video_path: Union[str, int]) -> ForestPlan:
        """Process video and generate ForestPlan.

        Args:
            video_path: Path to video file, stream URL, or camera index (int or string)
        """
        logger.info(f"Starting video analysis: {video_path}")

        # Initialize video reader
        reader = VideoReader(video_path)
        frame_results = []

        frame_count = 0
        processed_count = 0

        try:
            for frame_number, timestamp, frame in reader.read_frames(
                interval_seconds=self.config.frame_interval
            ):
                if self.config.max_frames and processed_count >= self.config.max_frames:
                    break

                # Analyze frame
                result = self.analyzer.analyze_frame(frame, frame_number, timestamp)
                frame_results.append(result)

                # Save debug frame if enabled
                if self.config.save_debug_frames:
                    self._save_debug_frame(frame, result, frame_number)

                processed_count += 1
                frame_count = frame_number

            logger.info(f"Processed {processed_count} frames from video")

            # Generate ForestPlan and aggregated statistics
            forest_plan, self.aggregated_stats = self.plan_generator.generate_plan(video_path, frame_results)

            # Save results
            self._save_results(forest_plan, frame_results, video_path)

            return forest_plan

        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            raise
        finally:
            reader.release()

    def _save_debug_frame(self, frame: np.ndarray, result: FrameAnalysisResult,
                         frame_number: int):
        """Save annotated debug frame."""
        debug_frame = frame.copy()

        # Draw detections
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            color = (0, 255, 0)  # Green
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

            # Add label
            label = f"{detection.species or 'tree'} {detection.confidence:.2f}"
            cv2.putText(debug_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add frame info
        info_text = f"Frame {frame_number}: {result.total_trees} trees, " \
                   f"Canopy: {result.canopy_coverage:.1%}"
        cv2.putText(debug_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save
        output_path = Path(self.config.output_dir) / f"debug_frame_{frame_number:04d}.jpg"
        cv2.imwrite(str(output_path), debug_frame)

    def _save_results(self, forest_plan: ForestPlan,
                     frame_results: List[FrameAnalysisResult],
                     video_path: Union[str, int]):
        """Save analysis results."""
        output_dir = Path(self.config.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save ForestPlan JSON
        plan_path = output_dir / f"forest_plan_{timestamp}.json"
        with open(plan_path, 'w') as f:
            f.write(forest_plan.to_json(indent=2))
        logger.info(f"ForestPlan saved to: {plan_path}")

        # Save frame results summary
        if self.config.save_intermediate_results:
            summary_path = output_dir / f"frame_results_{timestamp}.json"
            summary_data = {
                'video_path': str(video_path),  # Convert to string for JSON serialization
                'total_frames': len(frame_results),
                'config': asdict(self.config),
                'frames': [
                    {
                        'frame_number': r.frame_number,
                        'timestamp': r.timestamp,
                        'total_trees': r.total_trees,
                        'canopy_coverage': r.canopy_coverage,
                        'vegetation_density': r.vegetation_density,
                        'dominant_species': r.dominant_species,
                        'average_tree_height': r.average_tree_height
                    }
                    for r in frame_results
                ]
            }
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            logger.info(f"Frame results saved to: {summary_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point for the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Forest Analysis Pipeline v3.0"
    )
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['fast', 'standard', 'detailed', 'research'],
                       help='Analysis mode')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--interval', type=int, default=3,
                       help='Frame interval in seconds')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--use-sam2', action='store_true',
                       help='Enable SAM2 segmentation')
    parser.add_argument('--use-deepforest', action='store_true',
                       help='Enable DeepForest validation')
    parser.add_argument('--use-vit', action='store_true',
                       help='Enable ViT species classification')
    parser.add_argument('--debug', action='store_true',
                       help='Save debug frames')

    args = parser.parse_args()

    # Create configuration
    config = PipelineConfig(
        mode=AnalysisMode(args.mode),
        output_dir=args.output,
        frame_interval=args.interval,
        max_frames=args.max_frames,
        use_sam2=args.use_sam2,
        use_deepforest=args.use_deepforest,
        use_vit=args.use_vit,
        save_debug_frames=args.debug
    )

    # Run pipeline
    pipeline = EnhancedForestAnalysisPipeline(config)
    forest_plan = pipeline.process_video(args.video)

    # Print summary
    print("\n" + "="*80)
    print("FOREST ANALYSIS COMPLETE")
    print("="*80)
    print(json.dumps(forest_plan.summary(), indent=2))
    print("="*80)


if __name__ == "__main__":
    main()

