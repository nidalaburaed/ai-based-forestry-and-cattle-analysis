from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import date
import uuid
import json

# tree_struct.py


@dataclass
class Site:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = ""  # e.g., "grove", "barren", "wetland"
    nutrient_level: Optional[str] = None  # e.g., "low", "medium", "high"
    timber_potential: Optional[str] = None  # e.g., "low", "medium", "high"
    notes: Optional[str] = None


@dataclass
class Stand:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    species: List[str] = field(default_factory=list)  # dominant species
    age: Optional[int] = None  # years
    size_ha: Optional[float] = None  # hectares
    development_class: Optional[str] = None  # e.g., "seedling", "sapling", "mature"
    notes: Optional[str] = None


@dataclass
class ManagementActivity:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    responsible_party: Optional[str] = None
    harvest_estimate_m3: Optional[float] = None  # if harvesting activity
    notes: Optional[str] = None


@dataclass
class ValuableSite:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    protection_status: Optional[str] = None  # e.g., "protected", "candidate"
    recreational_value: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None  # {'lat': .., 'lon': ..}
    notes: Optional[str] = None


@dataclass
class MapLayer:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: Optional[str] = None
    url: Optional[str] = None  # link to map service or file
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EconomicInfo:
    estimated_value_eur: Optional[float] = None
    timber_sales_potential_m3: Optional[float] = None
    notes: Optional[str] = None


class ForestPlan:
    """
    Container for a forest plan:
    - sites: nutrient & timber potential areas
    - stands: species / age / development info
    - management_activities: planned actions & harvests
    - valuable_sites: protected/recreational sites
    - maps: thematic or plot-level map references
    - economic_info: financial / timber value estimates
    """

    def __init__(self, name: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.sites: List[Site] = []
        self.stands: List[Stand] = []
        self.management_activities: List[ManagementActivity] = []
        self.valuable_sites: List[ValuableSite] = []
        self.maps: List[MapLayer] = []
        self.economic_info: Optional[EconomicInfo] = None
        self.created_at = date.today()

    # --- adders ---
    def add_site(self, site: Site) -> Site:
        self.sites.append(site)
        return site

    def add_stand(self, stand: Stand) -> Stand:
        self.stands.append(stand)
        return stand

    def add_management_activity(self, activity: ManagementActivity) -> ManagementActivity:
        self.management_activities.append(activity)
        return activity

    def add_valuable_site(self, vsite: ValuableSite) -> ValuableSite:
        self.valuable_sites.append(vsite)
        return vsite

    def add_map(self, map_layer: MapLayer) -> MapLayer:
        self.maps.append(map_layer)
        return map_layer

    def set_economic_info(self, econ: EconomicInfo) -> EconomicInfo:
        self.economic_info = econ
        return econ

    # --- helpers ---
    def schedule_harvest(self, description: str, start: date, estimate_m3: Optional[float] = None,
                         end: Optional[date] = None, responsible: Optional[str] = None) -> ManagementActivity:
        act = ManagementActivity(
            description=description,
            start_date=start,
            end_date=end,
            harvest_estimate_m3=estimate_m3,
            responsible_party=responsible,
        )
        return self.add_management_activity(act)

    def find_sites_by_type(self, type_name: str) -> List[Site]:
        return [s for s in self.sites if s.type and s.type.lower() == type_name.lower()]

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": str(self.created_at),
            "counts": {
                "sites": len(self.sites),
                "stands": len(self.stands),
                "management_activities": len(self.management_activities),
                "valuable_sites": len(self.valuable_sites),
                "maps": len(self.maps),
            },
            "economic_info": asdict(self.economic_info) if self.economic_info else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": str(self.created_at),
            "sites": [asdict(s) for s in self.sites],
            "stands": [asdict(s) for s in self.stands],
            "management_activities": [asdict(a) for a in self.management_activities],
            "valuable_sites": [asdict(v) for v in self.valuable_sites],
            "maps": [asdict(m) for m in self.maps],
            "economic_info": asdict(self.economic_info) if self.economic_info else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # --- NLS (National Land Survey) map link helper ---
    @staticmethod
    def build_nls_map_link(lat: float, lon: float, zoom: int = 12) -> str:
        """
        Build a link to the National Land Survey map service (Karttapaikka).
        Example template:
          https://karttapaikka.fi/?lang=en&x={lon}&y={lat}&z={zoom}
        Note: different NLS endpoints may use different parameters; adjust if needed.
        """
        return f"https://karttapaikka.fi/?lang=en&x={lon}&y={lat}&z={zoom}"

    def __repr__(self) -> str:
        return f"<ForestPlan {self.name!r} id={self.id} sites={len(self.sites)} stands={len(self.stands)}>"

# Example usage (for quick testing; remove or comment out when importing as a module):
if __name__ == "__main__":
    fp = ForestPlan("Example Forest")
    fp.add_site(Site(name="North Grove", type="grove", nutrient_level="high", timber_potential="high"))
    fp.add_stand(Stand(species=["Pine", "Spruce"], age=40, size_ha=12.5, development_class="mature"))
    fp.schedule_harvest("Selective thinning", start=date(2026, 4, 1), estimate_m3=150.0, responsible="Forest Manager")
    fp.add_valuable_site(ValuableSite(name="Wet Meadow", protection_status="candidate", recreational_value="high",
                                      coordinates={"lat": 60.192059, "lon": 24.945831}))
    print(fp.summary())
    print("NLS map:", ForestPlan.build_nls_map_link(60.192059, 24.945831, zoom=14))