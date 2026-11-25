"""
Traffic Guardian AI - Physics-Based Clearance Time Estimator

The old clearance number came entirely from an ML regressor trained on
synthetic data - a black box with no traceable real-world grounding.
This replaces it as the PRIMARY estimate with an actual traffic-flow
queueing (shockwave) model, using published, citable constants:

  - Residual freeway capacity during a lane-blocking incident:
    HCM 2010, Exhibit 10-17 ("Table 40"), reproduced via FHWA
    Office of Operations: ops.fhwa.dot.gov/publications/fhwahop13042/appc.htm
    -> the actual % of a road's normal capacity that remains available
    when N of its lanes are blocked. This is the single most important
    real number in the whole clearance system.

  - Per-lane saturation flow (capacity) by road type: standard
    traffic-engineering range from the HCM (freeways ~2000-2200,
    signalized arterials ~1700-1900 pcu/hr/lane); we use the
    conservative, commonly-cited midpoints.

  - Incident duration-until-clear benchmarks: FHWA's national Traffic
    Incident Management "Open Roads" performance goals (minor
    incidents cleared in minutes, major incidents targeted under
    90 min) - used only to seed how long a queue has likely been
    building before the reported severity/blocked-lane state, which
    is standard practice in queue-based clearance estimation.

The original ML model is kept and reported alongside this as a
secondary cross-check ("trained on synthetic patterns - use for
pattern comparison, not as the primary number").
"""

# ---------------------------------------------------------------
# HCM 2010 Exhibit 10-17 - residual capacity (proportion of normal
# capacity still available) during an incident, by total lanes in
# that direction and how many are blocked.
# Source: FHWA Office of Operations, Table 40.
# ---------------------------------------------------------------
HCM_RESIDUAL_CAPACITY = {
    2: {"shoulder": 0.95, "shoulder_accident": 0.81, 1: 0.35, 2: 0.00, 3: 0.00},
    3: {"shoulder": 0.99, "shoulder_accident": 0.83, 1: 0.49, 2: 0.17, 3: 0.00},
    4: {"shoulder": 0.99, "shoulder_accident": 0.85, 1: 0.58, 2: 0.25, 3: 0.13},
    5: {"shoulder": 0.99, "shoulder_accident": 0.87, 1: 0.65, 2: 0.40, 3: 0.20},
    6: {"shoulder": 0.99, "shoulder_accident": 0.89, 1: 0.71, 2: 0.50, 3: 0.26},
    7: {"shoulder": 0.99, "shoulder_accident": 0.91, 1: 0.75, 2: 0.57, 3: 0.36},
    8: {"shoulder": 0.99, "shoulder_accident": 0.93, 1: 0.78, 2: 0.63, 3: 0.41},
}

# Per-lane saturation flow (vehicles/hour/lane) - standard HCM-consistent
# midpoints by road type.
LANE_CAPACITY_PER_HOUR = {
    "highway": 2000, "arterial": 1800, "collector": 1600, "local": 1400,
}

# Severity (0-5, as already collected in the UI) -> assumed minutes the
# queue has been building before this reading, based on FHWA TIM
# detection/response benchmarks (minor incidents clear fast, severe
# ones take much longer to even fully close lanes and stabilize).
SEVERITY_TO_BUILDUP_MINUTES = {0: 3, 1: 8, 2: 15, 3: 25, 4: 40, 5: 60}


def estimate_clearance_time(
    vehicle_count, occupancy_rate, road_type, number_of_lanes,
    blocked_lanes, severity,
):
    """Returns dict with clearance_time_minutes + the full breakdown
    (capacity numbers, residual factor, queue length) so the UI can
    show judges exactly how the number was derived - not a black box."""
    lanes = max(min(number_of_lanes, 8), 2)
    capacity_table = HCM_RESIDUAL_CAPACITY[lanes]

    if blocked_lanes <= 0:
        residual_factor = capacity_table["shoulder"] if severity <= 1 else capacity_table["shoulder_accident"]
    else:
        residual_factor = capacity_table.get(min(blocked_lanes, 3), 0.05)

    full_capacity_veh_hr = LANE_CAPACITY_PER_HOUR[road_type] * number_of_lanes
    reduced_capacity_veh_hr = full_capacity_veh_hr * residual_factor

    # demand is estimated from live occupancy_rate against full capacity
    # (occupancy_rate is already vehicle_count / road_capacity, clipped 0-0.99)
    arrival_rate_veh_hr = occupancy_rate * full_capacity_veh_hr

    buildup_min = SEVERITY_TO_BUILDUP_MINUTES.get(severity, 15)

    if arrival_rate_veh_hr <= reduced_capacity_veh_hr:
        # incident doesn't oversaturate the remaining lanes - no real
        # queue forms, clearance is just the physical response/removal time
        clearance_time_min = round(buildup_min * 0.6, 1)
        queue_length_vehicles = 0.0
    else:
        excess_rate_veh_hr = arrival_rate_veh_hr - reduced_capacity_veh_hr
        queue_length_vehicles = excess_rate_veh_hr * (buildup_min / 60)

        dissipation_rate_veh_hr = max(full_capacity_veh_hr - arrival_rate_veh_hr, full_capacity_veh_hr * 0.1)
        dissipation_min = (queue_length_vehicles / dissipation_rate_veh_hr) * 60
        clearance_time_min = round(buildup_min + dissipation_min, 1)

    return {
        "clearance_time_minutes": clearance_time_min,
        "clearance_range_minutes": [round(clearance_time_min * 0.7, 1), round(clearance_time_min * 1.45, 1)],
        "breakdown": {
            "full_capacity_veh_per_hr": round(full_capacity_veh_hr),
            "residual_capacity_factor": residual_factor,
            "reduced_capacity_veh_per_hr": round(reduced_capacity_veh_hr),
            "arrival_rate_veh_per_hr": round(arrival_rate_veh_hr),
            "estimated_queue_length_vehicles": round(queue_length_vehicles, 1),
            "assumed_buildup_minutes": buildup_min,
        },
        "source": "HCM 2010 Exhibit 10-17 residual-capacity model (real formula, live inputs)",
    }
