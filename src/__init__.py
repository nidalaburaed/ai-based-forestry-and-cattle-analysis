"""
Envaire Analysis Pipeline
==========================

A state-of-the-art AI-powered analysis system for:
- Forest inventory and management
- Cattle monitoring and herd management
"""

__version__ = "3.2.0"
__author__ = "Adrian"

from .pipeline import (
    EnhancedForestAnalysisPipeline,
    PipelineConfig,
    AnalysisMode,
    ModelBackend
)

from .data_structures import (
    ForestPlan,
    Site,
    Stand,
    ManagementActivity,
    ValuableSite,
    MapLayer,
    EconomicInfo
)

from .cattle_pipeline import CattleAnalysisPipeline

from .cattle_structures import (
    CattlePlan,
    CattleDetection,
    FrameCattleAnalysis,
    CattleTrack,
    HerdStatistics,
    CattleBehavior,
    CattleHealthStatus
)

from .report_generator import ReportGenerator

__all__ = [
    # Forest Analysis
    'EnhancedForestAnalysisPipeline',
    'PipelineConfig',
    'AnalysisMode',
    'ModelBackend',
    'ForestPlan',
    'Site',
    'Stand',
    'ManagementActivity',
    'ValuableSite',
    'MapLayer',
    'EconomicInfo',
    # Cattle Analysis
    'CattleAnalysisPipeline',
    'CattlePlan',
    'CattleDetection',
    'FrameCattleAnalysis',
    'CattleTrack',
    'HerdStatistics',
    'CattleBehavior',
    'CattleHealthStatus',
    # Reports
    'ReportGenerator',
]

