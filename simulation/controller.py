import traci
import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONFIGURATION ---
SUMO_BINARY = "sumo-gui" 
CONFIG_FILE = "traffic.sumocfg"
AMBULANCE_ID = "amb_1"

# Traffic Light Phases
PHASE_NS_GREEN = 0
PHASE_EW_GREEN = 2

# Path to the shared data folder
# simulation/controller.py -> simulation/ -> ROOT -> live_data
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_ROOT, "live_data", "traffic_stats.json")

def write_stats(step, cars_n, cars_s, cars_e, cars_w, current_phase, ambulance_active):
    """Writes live traffic data to a JSON file."""
    try:
        data = {
            "timestamp": step, # Using simulation step as timestamp
            "waiting_cars": {
                "north": cars_n, "south": cars_s, "east": cars_e, "west": cars_w
            },
            # Simple logic: if Phase is 0, Green is N/S. Otherwise E/W.
            "current_green_phase": "North/South" if current_phase == PHASE_NS_GREEN else "East/West",
            "ambulance_active": ambulance_active,
            "avg_wait_time": round((cars_n + cars_s + cars_e + cars_w) * 1.5, 1) # Fake estimate for demo
        }
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error writing JSON: {e}")

def run_logic():
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep() 

        # 1. READ SENSORS
        cars_n = traci.inductionloop.getLastStepVehicleNumber("sensor_north")
        cars_s = traci.inductionloop.getLastStepVehicleNumber("sensor_south")
        cars_e = traci.inductionloop.getLastStepVehicleNumber("sensor_east")
        cars_w = traci.inductionloop.getLastStepVehicleNumber("sensor_west")

        # 2. CHECK AMBULANCE
        ambulance_active = False
        active_vehicles = traci.vehicle.getIDList()
        if AMBULANCE_ID in active_vehicles:
            amb_lane = traci.vehicle.getLaneID(AMBULANCE_ID)
            if amb_lane and not amb_lane.startswith(":"):
                ambulance_active = True
                print(f"🚨 AMBULANCE DETECTED on {amb_lane}")
                
                # Priority Logic
                if "north_in" in amb_lane or "south_in" in amb_lane:
                    if traci.trafficlight.getPhase("junction_center") != PHASE_NS_GREEN:
                        traci.trafficlight.setPhase("junction_center", PHASE_NS_GREEN)
                elif "east_in" in amb_lane or "west_in" in amb_lane:
                    if traci.trafficlight.getPhase("junction_center") != PHASE_EW_GREEN:
                        traci.trafficlight.setPhase("junction_center", PHASE_EW_GREEN)

        # 3. WRITE DATA TO JSON
        current_phase = traci.trafficlight.getPhase("junction_center")
        write_stats(step, cars_n, cars_s, cars_e, cars_w, current_phase, ambulance_active)

        step += 1
    
    traci.close()

if __name__ == "__main__":
    sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE, "--start"]
    traci.start(sumo_cmd)
    run_logic()