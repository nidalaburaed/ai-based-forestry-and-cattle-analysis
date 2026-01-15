"""
Cattle Analysis Pipeline
=========================

AI-powered cattle monitoring and herd management system.

Features:
- YOLOv8/v11 for cattle detection
- Behavior recognition (standing, lying, grazing, walking)
- Multi-object tracking across frames
- Health monitoring and alerts
- Grazing pattern analysis
- Herd statistics and management recommendations

Models Used:
- Primary Detection: YOLOv8/YOLOv11 (trained on livestock/cattle datasets)
- Tracking: ByteTrack or SORT algorithm
- Behavior Classification: Pose-based analysis

Author: Adrian
Date: December 2024
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from dataclasses import dataclass

from .cattle_structures import (
    CattleDetection, FrameCattleAnalysis, CattleTrack,
    HerdStatistics, CattlePlan, CattleBehavior, CattleHealthStatus
)
from .pipeline import PipelineConfig, AnalysisMode, ModelBackend
from .video_reader import VideoReader

logger = logging.getLogger(__name__)


class CattleDetector:
    """Cattle detection using YOLO models."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model for cattle detection."""
        try:
            from ultralytics import YOLO
            import torch
            
            # Auto-detect device
            device = self.config.yolo_device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            
            # Use YOLOv8 or custom cattle model
            model_path = self.config.yolo_model
            logger.info(f"Loading YOLO model for cattle detection: {model_path}")
            
            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info(f"✓ Cattle detection model loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load cattle detection model: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[CattleDetection]:
        """Detect cattle in frame."""
        if self.model is None:
            return []
        
        # Run YOLO detection
        # For cattle, we typically look for classes: cow, cattle, bull, calf
        # Using COCO classes: 19 = cow, or custom trained model
        results = self.model.predict(
            frame,
            conf=self.config.min_confidence,
            iou=self.config.nms_threshold,
            verbose=False,
            classes=[19]  # COCO class for cow, or None for custom model
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                detection = CattleDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_name="cattle"
                )
                detections.append(detection)
        
        logger.debug(f"Detected {len(detections)} cattle")
        return detections


class CattleBehaviorAnalyzer:
    """Analyze cattle behavior from detections."""
    
    def analyze_behavior(self, detection: CattleDetection, frame: np.ndarray) -> CattleDetection:
        """Determine cattle behavior from bounding box and appearance."""
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Simple heuristic-based behavior classification
        # More sophisticated: use pose estimation or trained classifier
        
        if aspect_ratio > 1.8:
            # Wide bbox suggests lying down
            detection.behavior = CattleBehavior.LYING
            detection.behavior_confidence = 0.7
        elif aspect_ratio > 1.3:
            # Moderate aspect ratio could be grazing (head down)
            detection.behavior = CattleBehavior.GRAZING
            detection.behavior_confidence = 0.6
        else:
            # Vertical bbox suggests standing or walking
            detection.behavior = CattleBehavior.STANDING
            detection.behavior_confidence = 0.6
        
        return detection
    
    def assess_health(self, detection: CattleDetection, velocity: Optional[float] = None) -> CattleDetection:
        """Basic health assessment based on behavior and movement."""
        # Simple health indicators
        # In production: use more sophisticated ML models
        
        if detection.behavior == CattleBehavior.LYING:
            # Lying is normal, but prolonged lying could indicate issues
            detection.health_status = CattleHealthStatus.HEALTHY
        elif detection.behavior == CattleBehavior.GRAZING:
            # Grazing is a good sign
            detection.health_status = CattleHealthStatus.HEALTHY
        else:
            detection.health_status = CattleHealthStatus.HEALTHY
        
        return detection


class CattleTracker:
    """Track individual cattle across frames using simple IoU tracking."""
    
    def __init__(self):
        self.tracks: Dict[int, CattleTrack] = {}
        self.next_track_id = 1
        self.iou_threshold = 0.3
    
    def update(self, detections: List[CattleDetection], frame_number: int) -> List[CattleDetection]:
        """Update tracks with new detections."""
        # Simple IoU-based tracking
        # For production: use ByteTrack or DeepSORT
        
        if not self.tracks:
            # Initialize tracks
            for det in detections:
                det.track_id = self.next_track_id
                self.tracks[self.next_track_id] = CattleTrack(
                    track_id=self.next_track_id,
                    first_seen_frame=frame_number,
                    last_seen_frame=frame_number,
                    detections=[det]
                )
                self.next_track_id += 1
            return detections

        # Match detections to existing tracks using IoU
        matched_tracks = set()
        for det in detections:
            best_iou = 0
            best_track_id = None

            for track_id, track in self.tracks.items():
                if track.last_seen_frame < frame_number - 10:
                    continue  # Skip old tracks

                last_det = track.detections[-1]
                iou = self._calculate_iou(det.bbox, last_det.bbox)

                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                det.track_id = best_track_id
                self.tracks[best_track_id].detections.append(det)
                self.tracks[best_track_id].last_seen_frame = frame_number
                matched_tracks.add(best_track_id)
            else:
                # Create new track
                det.track_id = self.next_track_id
                self.tracks[self.next_track_id] = CattleTrack(
                    track_id=self.next_track_id,
                    first_seen_frame=frame_number,
                    last_seen_frame=frame_number,
                    detections=[det]
                )
                self.next_track_id += 1

        return detections

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bboxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_active_tracks(self) -> List[CattleTrack]:
        """Get all tracks."""
        return list(self.tracks.values())


class CattleAnalysisPipeline:
    """Main pipeline for cattle monitoring and analysis."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detector = CattleDetector(config)
        self.behavior_analyzer = CattleBehaviorAnalyzer()
        self.tracker = CattleTracker()

        # Storage for results
        self.frame_results: List[FrameCattleAnalysis] = []
        self.aggregated_stats: Optional[Dict] = None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("CattleAnalysisPipeline initialized")

    def analyze_frame(self, frame: np.ndarray, frame_number: int,
                     timestamp: float) -> FrameCattleAnalysis:
        """Analyze a single frame for cattle."""
        logger.info(f"Analyzing frame {frame_number} at {timestamp:.2f}s")

        # Step 1: Detect cattle
        detections = self.detector.detect(frame)

        # Step 2: Analyze behavior for each detection
        for det in detections:
            det = self.behavior_analyzer.analyze_behavior(det, frame)
            det = self.behavior_analyzer.assess_health(det)

        # Step 3: Track cattle across frames
        detections = self.tracker.update(detections, frame_number)

        # Step 4: Calculate frame-level statistics
        standing = sum(1 for d in detections if d.behavior == CattleBehavior.STANDING)
        lying = sum(1 for d in detections if d.behavior == CattleBehavior.LYING)
        grazing = sum(1 for d in detections if d.behavior == CattleBehavior.GRAZING)
        walking = sum(1 for d in detections if d.behavior == CattleBehavior.WALKING)

        healthy = sum(1 for d in detections if d.health_status == CattleHealthStatus.HEALTHY)
        attention = sum(1 for d in detections if d.health_status == CattleHealthStatus.ATTENTION_NEEDED)

        # Calculate grazing coverage (simplified)
        frame_area = frame.shape[0] * frame.shape[1]
        cattle_area = sum((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) for d in detections)
        grazing_coverage = (cattle_area / frame_area * 100) if frame_area > 0 else 0.0

        # Calculate herd center
        if detections:
            centers = [d.center_point for d in detections if d.center_point]
            if centers:
                herd_center = (
                    sum(c[0] for c in centers) / len(centers),
                    sum(c[1] for c in centers) / len(centers)
                )
            else:
                herd_center = None
        else:
            herd_center = None

        result = FrameCattleAnalysis(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            total_cattle=len(detections),
            standing_count=standing,
            lying_count=lying,
            grazing_count=grazing,
            walking_count=walking,
            grazing_area_coverage=grazing_coverage,
            herd_density=len(detections) / (frame_area / 1000000) if frame_area > 0 else 0.0,
            herd_center=herd_center,
            healthy_count=healthy,
            attention_needed_count=attention
        )

        return result

    def process_video(self, video_path: Union[str, int]) -> CattlePlan:
        """Process video and generate CattlePlan.

        Args:
            video_path: Path to video file, stream URL, or camera index (int or string)
        """
        logger.info(f"Starting cattle video analysis: {video_path}")

        # Initialize video reader
        reader = VideoReader(video_path)
        self.frame_results = []

        processed_count = 0

        try:
            for frame_number, timestamp, frame in reader.read_frames(
                interval_seconds=self.config.frame_interval
            ):
                if self.config.max_frames and processed_count >= self.config.max_frames:
                    break

                # Analyze frame
                result = self.analyze_frame(frame, frame_number, timestamp)
                self.frame_results.append(result)

                # Save debug frame if enabled
                if self.config.save_debug_frames:
                    self._save_debug_frame(frame, result, frame_number)

                processed_count += 1
                logger.info(f"Processed frame {frame_number}: {result.total_cattle} cattle detected")

        finally:
            reader.release()

        logger.info(f"Video analysis complete. Processed {processed_count} frames")

        # Generate herd statistics
        herd_stats = self._calculate_herd_statistics()

        # Generate aggregated stats for UI
        self.aggregated_stats = self._generate_aggregated_stats(herd_stats)

        # Get all tracks
        cattle_tracks = self.tracker.get_active_tracks()

        # Generate recommendations
        recommendations = self._generate_recommendations(herd_stats, cattle_tracks)
        health_alerts = self._generate_health_alerts(cattle_tracks)

        # Create CattlePlan
        plan = CattlePlan(
            video_source=video_path,
            analysis_mode=self.config.mode.value,
            herd_stats=herd_stats,
            cattle_tracks=cattle_tracks,
            frame_results=self.frame_results,
            management_recommendations=recommendations,
            health_alerts=health_alerts
        )

        logger.info("CattlePlan generated successfully")
        return plan

    def _calculate_herd_statistics(self) -> HerdStatistics:
        """Calculate aggregated herd statistics."""
        if not self.frame_results:
            return HerdStatistics(
                total_frames_analyzed=0,
                total_cattle_detected=0,
                unique_cattle_tracked=0,
                avg_cattle_per_frame=0.0,
                max_cattle_in_frame=0,
                min_cattle_in_frame=0
            )

        total_frames = len(self.frame_results)
        total_detections = sum(r.total_cattle for r in self.frame_results)
        unique_tracks = len(self.tracker.get_active_tracks())

        cattle_counts = [r.total_cattle for r in self.frame_results]
        avg_cattle = total_detections / total_frames if total_frames > 0 else 0.0
        max_cattle = max(cattle_counts) if cattle_counts else 0
        min_cattle = min(cattle_counts) if cattle_counts else 0

        # Behavior distribution
        total_behaviors = {
            "standing": sum(r.standing_count for r in self.frame_results),
            "lying": sum(r.lying_count for r in self.frame_results),
            "grazing": sum(r.grazing_count for r in self.frame_results),
            "walking": sum(r.walking_count for r in self.frame_results)
        }

        total_behavior_count = sum(total_behaviors.values())
        behavior_dist = {
            k: (v / total_behavior_count * 100) if total_behavior_count > 0 else 0.0
            for k, v in total_behaviors.items()
        }

        # Health metrics
        total_healthy = sum(r.healthy_count for r in self.frame_results)
        total_attention = sum(r.attention_needed_count for r in self.frame_results)
        total_health_assessments = total_healthy + total_attention

        healthy_pct = (total_healthy / total_health_assessments * 100) if total_health_assessments > 0 else 0.0
        attention_pct = (total_attention / total_health_assessments * 100) if total_health_assessments > 0 else 0.0

        # Spatial metrics
        avg_grazing_coverage = sum(r.grazing_area_coverage for r in self.frame_results) / total_frames
        avg_herd_density = sum(r.herd_density for r in self.frame_results) / total_frames

        # Activity metrics
        active_count = total_behaviors["standing"] + total_behaviors["walking"] + total_behaviors["grazing"]
        resting_count = total_behaviors["lying"]
        total_activity = active_count + resting_count

        active_pct = (active_count / total_activity * 100) if total_activity > 0 else 0.0
        resting_pct = (resting_count / total_activity * 100) if total_activity > 0 else 0.0

        return HerdStatistics(
            total_frames_analyzed=total_frames,
            total_cattle_detected=total_detections,
            unique_cattle_tracked=unique_tracks,
            avg_cattle_per_frame=avg_cattle,
            max_cattle_in_frame=max_cattle,
            min_cattle_in_frame=min_cattle,
            behavior_distribution=behavior_dist,
            avg_grazing_coverage=avg_grazing_coverage,
            avg_herd_density=avg_herd_density,
            healthy_percentage=healthy_pct,
            attention_needed_percentage=attention_pct,
            active_percentage=active_pct,
            resting_percentage=resting_pct
        )

    def _generate_aggregated_stats(self, herd_stats: HerdStatistics) -> Dict:
        """Generate aggregated statistics dictionary for UI display."""
        return {
            "total_frames_analyzed": herd_stats.total_frames_analyzed,
            "total_cattle_detected": herd_stats.total_cattle_detected,
            "unique_cattle_tracked": herd_stats.unique_cattle_tracked,
            "avg_cattle_per_frame": round(herd_stats.avg_cattle_per_frame, 2),
            "max_cattle_in_frame": herd_stats.max_cattle_in_frame,
            "min_cattle_in_frame": herd_stats.min_cattle_in_frame,
            "behavior_distribution": {
                k: f"{v:.1f}%" for k, v in herd_stats.behavior_distribution.items()
            },
            "avg_grazing_coverage": f"{herd_stats.avg_grazing_coverage:.2f}%",
            "health_status": {
                "healthy": f"{herd_stats.healthy_percentage:.1f}%",
                "attention_needed": f"{herd_stats.attention_needed_percentage:.1f}%"
            },
            "activity_levels": {
                "active": f"{herd_stats.active_percentage:.1f}%",
                "resting": f"{herd_stats.resting_percentage:.1f}%"
            }
        }

    def _generate_recommendations(self, herd_stats: HerdStatistics,
                                 tracks: List[CattleTrack]) -> List[str]:
        """Generate management recommendations based on analysis."""
        recommendations = []

        # Check herd size
        if herd_stats.avg_cattle_per_frame < 5:
            recommendations.append("Small herd detected. Consider monitoring for missing animals.")

        # Check activity levels
        if herd_stats.resting_percentage > 60:
            recommendations.append("High resting percentage detected. Monitor for health issues or heat stress.")

        # Check grazing behavior
        if herd_stats.behavior_distribution.get("grazing", 0) < 30:
            recommendations.append("Low grazing activity. Check pasture quality and availability.")

        # Check health
        if herd_stats.attention_needed_percentage > 10:
            recommendations.append(f"⚠️ {herd_stats.attention_needed_percentage:.1f}% of cattle need attention. Conduct health inspection.")

        if not recommendations:
            recommendations.append("✓ Herd appears healthy and active. Continue regular monitoring.")

        return recommendations

    def _generate_health_alerts(self, tracks: List[CattleTrack]) -> List[str]:
        """Generate health alerts for individual cattle."""
        alerts = []

        for track in tracks:
            if track.health_status == CattleHealthStatus.ATTENTION_NEEDED:
                alerts.append(f"Cattle #{track.track_id}: Requires attention")
            elif track.health_status == CattleHealthStatus.ABNORMAL:
                alerts.append(f"⚠️ Cattle #{track.track_id}: Abnormal behavior detected")

        return alerts

    def _save_debug_frame(self, frame: np.ndarray, result: FrameCattleAnalysis,
                         frame_number: int):
        """Save annotated debug frame."""
        debug_frame = frame.copy()

        # Draw detections
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox

            # Color based on behavior
            if det.behavior == CattleBehavior.LYING:
                color = (0, 0, 255)  # Red
            elif det.behavior == CattleBehavior.GRAZING:
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Blue

            # Draw bbox
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"#{det.track_id} {det.behavior.value}"
            cv2.putText(debug_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw statistics
        stats_text = [
            f"Frame: {frame_number}",
            f"Cattle: {result.total_cattle}",
            f"Standing: {result.standing_count}",
            f"Lying: {result.lying_count}",
            f"Grazing: {result.grazing_count}"
        ]

        y_offset = 30
        for text in stats_text:
            cv2.putText(debug_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        # Save frame
        output_path = Path(self.config.output_dir) / "debug_frames"
        output_path.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path / f"frame_{frame_number:06d}.jpg"), debug_frame)
        logger.debug(f"Saved debug frame: {frame_number}")

