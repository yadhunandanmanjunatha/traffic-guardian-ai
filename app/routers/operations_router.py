"""
Traffic Guardian AI - Operations & City Analytics Router

Provides city corridor health, simulated camera analytics, preset scenarios,
emergency green corridor calculations, and eco-impact estimations.
"""
from typing import Optional, List
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["operations"])


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class GreenCorridorRequest(BaseModel):
    source: str = Field(..., description="Starting landmark ID")
    destination: str = Field(..., description="Destination landmark ID")
    vehicle_type: str = Field("ambulance", description="ambulance | fire_engine | police")
    priority_level: int = Field(1, description="1 (critical) to 3 (urgent)")


class EcoImpactRequest(BaseModel):
    distance_km: float = Field(..., description="Route distance in km")
    travel_time_min: float = Field(..., description="Actual travel time in min")
    free_flow_time_min: float = Field(..., description="Free-flow travel time in min")
    congestion_class: str = Field("Moderate", description="Low, Moderate, High, Severe")
    daily_commuters: int = Field(5000, description="Estimated daily commuters on route")


# -----------------------------------------------------------------------------
# Curated Bangalore Scenarios
# -----------------------------------------------------------------------------
BANGALORE_SCENARIOS = [
    {
        "id": "silk_board_peak",
        "name": "Silk Board Monsoon Peak",
        "category": "Severe Bottleneck",
        "description": "Heavy Friday evening commute during drizzle at Silk Board Junction connecting BTM and HSR.",
        "badge": "Critical Rush",
        "params": {
            "car_count": 85,
            "bike_count": 65,
            "bus_count": 8,
            "truck_count": 4,
            "road_type": "arterial",
            "number_of_lanes": 4,
            "speed_limit": 50,
            "road_length_km": 1.8,
            "average_speed": 11,
            "hour": 18,
            "rainfall_mm": 5.5,
            "visibility_km": 6.0,
            "wind_speed": 14,
            "temperature": 23,
            "incident_exists": True,
            "severity": 3,
            "blocked_lanes": 1
        }
    },
    {
        "id": "orr_bellandur_morning",
        "name": "Bellandur Tech Corridor",
        "category": "Office Peak",
        "description": "Outer Ring Road morning rush towards Ecospace and Prestige Tech Park.",
        "badge": "High Congestion",
        "params": {
            "car_count": 68,
            "bike_count": 42,
            "bus_count": 6,
            "truck_count": 2,
            "road_type": "arterial",
            "number_of_lanes": 6,
            "speed_limit": 60,
            "road_length_km": 2.5,
            "average_speed": 22,
            "hour": 9,
            "rainfall_mm": 0,
            "visibility_km": 9.5,
            "wind_speed": 8,
            "temperature": 26,
            "incident_exists": False,
            "severity": 0,
            "blocked_lanes": 0
        }
    },
    {
        "id": "airport_express_night",
        "name": "Airport Expressway Night",
        "category": "Free Flow",
        "description": "Bellary Road Elevated Tollway towards Kempegowda International Airport at 11 PM.",
        "badge": "Smooth Travel",
        "params": {
            "car_count": 22,
            "bike_count": 6,
            "bus_count": 3,
            "truck_count": 5,
            "road_type": "highway",
            "number_of_lanes": 6,
            "speed_limit": 80,
            "road_length_km": 8.0,
            "average_speed": 74,
            "hour": 23,
            "rainfall_mm": 0,
            "visibility_km": 10.0,
            "wind_speed": 12,
            "temperature": 21,
            "incident_exists": False,
            "severity": 0,
            "blocked_lanes": 0
        }
    },
    {
        "id": "stadium_match_day",
        "name": "Chinnaswamy Match Surge",
        "category": "Event Surge",
        "description": "Cricket match release crowd on MG Road, Cubbon Road, and Kasturba Road.",
        "badge": "Surge Congestion",
        "params": {
            "car_count": 72,
            "bike_count": 58,
            "bus_count": 10,
            "truck_count": 1,
            "road_type": "arterial",
            "number_of_lanes": 4,
            "speed_limit": 50,
            "road_length_km": 1.2,
            "average_speed": 14,
            "hour": 21,
            "rainfall_mm": 1.0,
            "visibility_km": 8.0,
            "wind_speed": 7,
            "temperature": 25,
            "incident_exists": True,
            "severity": 2,
            "blocked_lanes": 1
        }
    },
    {
        "id": "sunday_indiranagar",
        "name": "Indiranagar 100ft Rd Leisure",
        "category": "Moderate Flow",
        "description": "Sunday afternoon dining and shopping traffic on 100 Feet Road.",
        "badge": "Moderate Flow",
        "params": {
            "car_count": 36,
            "bike_count": 25,
            "bus_count": 1,
            "truck_count": 0,
            "road_type": "collector",
            "number_of_lanes": 4,
            "speed_limit": 40,
            "road_length_km": 1.6,
            "average_speed": 27,
            "hour": 14,
            "rainfall_mm": 0,
            "visibility_km": 10.0,
            "wind_speed": 9,
            "temperature": 28,
            "incident_exists": False,
            "severity": 0,
            "blocked_lanes": 0
        }
    }
]


# -----------------------------------------------------------------------------
# City Corridors Data
# -----------------------------------------------------------------------------
CITY_CORRIDORS = [
    {
        "id": "orr_central",
        "name": "Outer Ring Road (Silk Board - Marathahalli)",
        "length_km": 11.2,
        "current_speed_kmh": 16.4,
        "free_flow_speed_kmh": 48.0,
        "ths_score": 38,
        "band": "Strained",
        "congestion_class": "Severe",
        "active_incidents": 2,
        "alert": "Waterlogging alert near Bellandur flyover service lane",
        "trend": "worsening"
    },
    {
        "id": "hosur_elevated",
        "name": "Hosur Road Expressway (Silk Board - Electronic City)",
        "length_km": 9.8,
        "current_speed_kmh": 58.0,
        "free_flow_speed_kmh": 70.0,
        "ths_score": 82,
        "band": "Optimal",
        "congestion_class": "Low",
        "active_incidents": 0,
        "alert": "Toll plaza flow smooth; FASTag lanes active",
        "trend": "steady"
    },
    {
        "id": "airport_corridor",
        "name": "Bellary Road (Hebbal - Yelahanka - KIA)",
        "length_km": 24.5,
        "current_speed_kmh": 52.0,
        "free_flow_speed_kmh": 75.0,
        "ths_score": 74,
        "band": "Good",
        "congestion_class": "Moderate",
        "active_incidents": 1,
        "alert": "Slow queue forming on Hebbal flyover entry ramp",
        "trend": "steady"
    },
    {
        "id": "old_madras_road",
        "name": "Old Madras Road (Tin Factory - KR Puram)",
        "length_km": 5.4,
        "current_speed_kmh": 12.0,
        "free_flow_speed_kmh": 45.0,
        "ths_score": 29,
        "band": "Critical",
        "congestion_class": "Severe",
        "active_incidents": 1,
        "alert": "Metro construction lane bottleneck near Benniganahalli",
        "trend": "worsening"
    },
    {
        "id": "central_cbd",
        "name": "CBD Loop (MG Road - Residency Rd - Richmond Rd)",
        "length_km": 4.2,
        "current_speed_kmh": 22.5,
        "free_flow_speed_kmh": 40.0,
        "ths_score": 64,
        "band": "Moderate",
        "congestion_class": "Moderate",
        "active_incidents": 0,
        "alert": "Moderate shopping queue near Brigade Road junction",
        "trend": "improving"
    },
    {
        "id": "sarjapur_road",
        "name": "Sarjapur Road (Koramangala - Carmelaram)",
        "length_km": 8.7,
        "current_speed_kmh": 18.2,
        "free_flow_speed_kmh": 42.0,
        "ths_score": 46,
        "band": "Strained",
        "congestion_class": "High",
        "active_incidents": 1,
        "alert": "Road widening excavation near Kaikondrahalli Lake",
        "trend": "steady"
    },
    {
        "id": "yeshwanthpur_peenya",
        "name": "Tumkur Road Industrial Arterial (Peenya - Yeshwanthpur)",
        "length_km": 7.6,
        "current_speed_kmh": 34.0,
        "free_flow_speed_kmh": 55.0,
        "ths_score": 68,
        "band": "Moderate",
        "congestion_class": "Moderate",
        "active_incidents": 0,
        "alert": "Truck freight movement orderly on grade level",
        "trend": "improving"
    },
    {
        "id": "bannerghatta_strip",
        "name": "Bannerghatta Road (Dairy Circle - Meenakshi Mall)",
        "length_km": 7.1,
        "current_speed_kmh": 15.1,
        "free_flow_speed_kmh": 40.0,
        "ths_score": 34,
        "band": "Critical",
        "congestion_class": "Severe",
        "active_incidents": 1,
        "alert": "Heavy slow-down near Sagar Hospital junction",
        "trend": "worsening"
    }
]


# -----------------------------------------------------------------------------
# City Alerts & Police Bulletins
# -----------------------------------------------------------------------------
CITY_ALERTS = [
    {
        "id": "alt_01",
        "time": "Just now",
        "level": "urgent",
        "title": "Waterlogging Advisory — Bellandur Ecospace",
        "message": "Water accumulation on ORR service road towards Marathahalli. 1 lane restricted. Traffic police deployed for manual signaling.",
        "location": "Bellandur Outer Ring Road"
    },
    {
        "id": "alt_02",
        "time": "12m ago",
        "level": "warning",
        "title": "Hebbal Flyover Entry Bottleneck",
        "message": "Heavy merging queue from Airport Road down-ramp onto ORR. Estimated transit delay: +14 mins.",
        "location": "Hebbal Junction"
    },
    {
        "id": "alt_03",
        "time": "28m ago",
        "level": "info",
        "title": "Green Corridor Cleared — St. John's to Manipal",
        "message": "Organ transport ambulance successfully escorted across Silk Board and Koramangala. 10 traffic signals synchronized.",
        "location": "Hosur Rd / Intermediate Ring Rd"
    },
    {
        "id": "alt_04",
        "time": "45m ago",
        "level": "info",
        "title": "Metro Line 2B Girder Launch",
        "message": "Night barricading scheduled on Airport Link near Chikkajala between 23:00 and 05:00. Commuters advised to use bypass.",
        "location": "Bellary Road"
    }
]


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@router.get("/api/scenarios")
def get_scenarios():
    """Returns curated Bangalore traffic presets for instant testing."""
    return {"scenarios": BANGALORE_SCENARIOS}


@router.get("/api/corridors")
def get_corridors():
    """Returns real-time health scores and status for major Bangalore corridors."""
    avg_speed = sum(c["current_speed_kmh"] for c in CITY_CORRIDORS) / len(CITY_CORRIDORS)
    avg_ths = sum(c["ths_score"] for c in CITY_CORRIDORS) / len(CITY_CORRIDORS)
    total_incidents = sum(c["active_incidents"] for c in CITY_CORRIDORS)
    
    return {
        "city": "Bengaluru",
        "network_ths": round(avg_ths, 1),
        "network_avg_speed_kmh": round(avg_speed, 1),
        "active_incident_count": total_incidents,
        "corridors": CITY_CORRIDORS
    }


@router.get("/api/city-alerts")
def get_city_alerts():
    """Returns real-time traffic alerts and dispatch advisories."""
    return {"alerts": CITY_ALERTS}


@router.post("/api/green-corridor")
def compute_green_corridor(req: GreenCorridorRequest):
    """
    Emergency Vehicle (Ambulance / Fire) Route Optimization.
    Calculates signal priority override clearance, time saved, and escort protocol.
    """
    # Sample lookup coordinates and approximate distances
    estimated_distance_km = 8.5
    normal_delay_min = 28.0
    
    # Priority level 1 gives preemptive green wave (approx 55% reduction in time)
    reduction_pct = 0.58 if req.priority_level == 1 else 0.42
    optimized_time_min = round(normal_delay_min * (1 - reduction_pct), 1)
    time_saved_min = round(normal_delay_min - optimized_time_min, 1)
    signal_count = max(4, int(estimated_distance_km * 1.4))
    
    return {
        "emergency_id": f"GC-BLR-{req.source[:3].upper()}-{req.destination[:3].upper()}",
        "vehicle_type": req.vehicle_type,
        "priority_level": req.priority_level,
        "source": req.source,
        "destination": req.destination,
        "distance_km": estimated_distance_km,
        "standard_travel_time_min": normal_delay_min,
        "emergency_green_wave_time_min": optimized_time_min,
        "time_saved_min": time_saved_min,
        "signals_synchronized": signal_count,
        "interventions": [
            f"Preemptive green wave active across {signal_count} traffic junctions",
            "Left-turn dedicated lane clearance alert transmitted to city control",
            "Hospital ER trauma team notification dispatched with live GPS ETA",
            "Variable Message Signs (VMS) along corridor switched to 'Yield to Emergency Vehicle'"
        ],
        "status": "CORRIDOR_ARMED"
    }


@router.post("/api/eco-impact")
def compute_eco_impact(req: EcoImpactRequest):
    """
    Computes environmental footprint (excess fuel burned and CO2 emissions)
    caused by congestion delay compared to free-flow conditions.
    """
    delay_min = max(0.0, req.travel_time_min - req.free_flow_time_min)
    
    # Typical passenger car consumes approx 0.8 to 1.2 litres/hr during idling/stop-and-go
    fuel_wasted_per_veh_litres = (delay_min / 60.0) * 1.15
    # Gasoline produces approx 2.31 kg CO2 per litre burned
    co2_excess_per_veh_kg = fuel_wasted_per_veh_litres * 2.31
    
    # Fuel cost in Bangalore (approx ₹102/litre)
    fuel_cost_per_litre_inr = 102.86
    individual_cost_inr = fuel_wasted_per_veh_litres * fuel_cost_per_litre_inr
    
    # Corridor daily aggregate for commuter volume
    corridor_fuel_litres = fuel_wasted_per_veh_litres * req.daily_commuters
    corridor_co2_tonnes = (co2_excess_per_veh_kg * req.daily_commuters) / 1000.0
    corridor_cost_inr = individual_cost_inr * req.daily_commuters
    
    # Eco grade
    if delay_min < 4:
        eco_grade = "A (Eco Friendly)"
        eco_color = "#10B981"
    elif delay_min < 12:
        eco_grade = "B (Moderate Impact)"
        eco_color = "#F59E0B"
    elif delay_min < 22:
        eco_grade = "C (High Emissions)"
        eco_color = "#EA580C"
    else:
        eco_grade = "D (Severe Environmental Waste)"
        eco_color = "#EF4444"

    return {
        "delay_minutes": round(delay_min, 1),
        "eco_grade": eco_grade,
        "eco_color": eco_color,
        "per_vehicle": {
            "fuel_wasted_litres": round(fuel_wasted_per_veh_litres, 2),
            "co2_excess_kg": round(co2_excess_per_veh_kg, 2),
            "money_wasted_inr": round(individual_cost_inr, 1)
        },
        "daily_corridor_aggregate": {
            "commuters_counted": req.daily_commuters,
            "fuel_wasted_litres": round(corridor_fuel_litres, 0),
            "co2_excess_tonnes": round(corridor_co2_tonnes, 2),
            "economic_loss_inr": round(corridor_cost_inr, 0)
        },
        "green_advisory": (
            "Car-pooling or taking Namma Metro Purple/Green lines on this corridor "
            f"can save up to {round(corridor_co2_tonnes * 0.7, 1)} tonnes of CO2 daily."
        )
    }
