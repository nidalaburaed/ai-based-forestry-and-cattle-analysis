"""
Cattle Analysis Data Structures
================================

Data structures for cattle detection, tracking, and herd management.

Author: Adrian
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime


class CattleBehavior(Enum):
    """Cattle behavior states."""
    STANDING = "standing"
    LYING = "lying"
    WALKING = "walking"
    GRAZING = "grazing"
    RUNNING = "running"
    UNKNOWN = "unknown"


class CattleHealthStatus(Enum):
    """Cattle health indicators."""
    HEALTHY = "healthy"
    ATTENTION_NEEDED = "attention_needed"
    ABNORMAL = "abnormal"
    UNKNOWN = "unknown"


@dataclass
class CattleDetection:
    """Single cattle detection in a frame."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str = "cattle"
    track_id: Optional[int] = None  # For multi-frame tracking
    
    # Behavior analysis
    behavior: CattleBehavior = CattleBehavior.UNKNOWN
    behavior_confidence: float = 0.0
    
    # Health indicators
    health_status: CattleHealthStatus = CattleHealthStatus.UNKNOWN
    
    # Physical metrics
    estimated_weight_kg: Optional[float] = None
    body_condition_score: Optional[float] = None  # 1-5 scale
    
    # Position and movement
    center_point: Optional[Tuple[float, float]] = None
    velocity: Optional[float] = None  # pixels per second
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate center point from bbox."""
        if self.center_point is None:
            x1, y1, x2, y2 = self.bbox
            self.center_point = ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class FrameCattleAnalysis:
    """Results from analyzing cattle in a single frame."""
    frame_number: int
    timestamp: float
    detections: List[CattleDetection]
    
    # Frame-level statistics
    total_cattle: int
    standing_count: int = 0
    lying_count: int = 0
    grazing_count: int = 0
    walking_count: int = 0
    
    # Spatial analysis
    grazing_area_coverage: float = 0.0  # Percentage of frame
    herd_density: float = 0.0  # Cattle per unit area
    herd_center: Optional[Tuple[float, float]] = None
    
    # Health indicators
    healthy_count: int = 0
    attention_needed_count: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CattleTrack:
    """Track of individual cattle across multiple frames."""
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    detections: List[CattleDetection] = field(default_factory=list)
    
    # Behavior over time
    dominant_behavior: CattleBehavior = CattleBehavior.UNKNOWN
    behavior_history: List[CattleBehavior] = field(default_factory=list)
    
    # Movement analysis
    total_distance_traveled: float = 0.0
    average_velocity: float = 0.0
    path_points: List[Tuple[float, float]] = field(default_factory=list)
    
    # Health tracking
    health_status: CattleHealthStatus = CattleHealthStatus.UNKNOWN
    health_alerts: List[str] = field(default_factory=list)


@dataclass
class HerdStatistics:
    """Aggregated statistics for the entire herd."""
    total_frames_analyzed: int
    total_cattle_detected: int
    unique_cattle_tracked: int
    
    # Average counts per frame
    avg_cattle_per_frame: float
    max_cattle_in_frame: int
    min_cattle_in_frame: int
    
    # Behavior distribution
    behavior_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Spatial metrics
    avg_grazing_coverage: float = 0.0
    avg_herd_density: float = 0.0
    
    # Health metrics
    healthy_percentage: float = 0.0
    attention_needed_percentage: float = 0.0
    
    # Activity metrics
    avg_movement_velocity: float = 0.0
    active_percentage: float = 0.0  # Standing/walking/grazing
    resting_percentage: float = 0.0  # Lying
    
    # Time-based analysis
    analysis_duration_seconds: float = 0.0
    video_duration_seconds: float = 0.0


@dataclass
class CattlePlan:
    """Complete cattle monitoring and management plan."""
    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    video_source: str = ""
    analysis_mode: str = "standard"
    
    # Herd statistics
    herd_stats: Optional[HerdStatistics] = None
    
    # Individual tracks
    cattle_tracks: List[CattleTrack] = field(default_factory=list)
    
    # Frame-by-frame results
    frame_results: List[FrameCattleAnalysis] = field(default_factory=list)
    
    # Recommendations
    management_recommendations: List[str] = field(default_factory=list)
    health_alerts: List[str] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary dictionary."""
        return {
            "generated_at": self.generated_at,
            "video_source": self.video_source,
            "herd_statistics": {
                "total_cattle": self.herd_stats.total_cattle_detected if self.herd_stats else 0,
                "unique_tracked": self.herd_stats.unique_cattle_tracked if self.herd_stats else 0,
                "avg_per_frame": self.herd_stats.avg_cattle_per_frame if self.herd_stats else 0,
                "behavior_distribution": self.herd_stats.behavior_distribution if self.herd_stats else {},
                "health_status": {
                    "healthy": f"{self.herd_stats.healthy_percentage:.1f}%" if self.herd_stats else "0%",
                    "attention_needed": f"{self.herd_stats.attention_needed_percentage:.1f}%" if self.herd_stats else "0%"
                }
            },
            "recommendations": self.management_recommendations,
            "health_alerts": self.health_alerts
        }

