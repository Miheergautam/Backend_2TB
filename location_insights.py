import requests

from utils import call2_deepseek

def find_location_parameters(road_location):
    terrain_response = call2_deepseek(
        "You are a geographical analyst. Provide the terrain type in 1-2 lines.",
        f"What is the terrain type at {road_location}? Respond with the terrain type."
    )
    terrain = terrain_response.strip()
    print(terrain)

    # 2. Climate and Working Season
    climate_response = call2_deepseek(
        "You are a geographical analyst. Provide the climate zone and the working season for road construction in 1-2 line.",
        f"What is the climate zone and suitable working season at {road_location}? Respond with the climate type and working season."
    )
    climate = climate_response.strip()
    print(climate)

    # 3. Logistical Difficulty
    logistics_response = call2_deepseek(
        "You are a geographical analyst. Provide 1-2 lines on logistical difficulty.",
        f"What is the logistical difficulty for construction at {road_location}? Respond with the difficulty."
    )
    logistics = logistics_response.strip()
    print(logistics)

    # 4. Safety and Threats
    safety_response = call2_deepseek(
        "You are a geographical analyst. Provide the human safety/terrorism critique in 1-2 lines.",
        f"What is the human threat/terrorism risk at {road_location}? Respond with the risk level."
    )
    safety = safety_response.strip()
    print(safety)

    # 5. Soil Type and Rock Availability
    soil_response = call2_deepseek(
        "You are a geographical analyst. Provide the soil type and rock availability in 1-2 line.",
        f"What is the soil type and are rocks available for aggregates at {road_location}? Respond with this information."
    )
    soil_type = soil_response.strip()
    print(soil_type)

    # 6. Material Availability
    materials_response = call2_deepseek(
        "You are a geographical analyst. Provide fuel/cement vendor availability nearby in 1-2 line.",
        f"Are diesel/petrol pumps and cement vendors available near {road_location}? Respond with this availability info."
    )
    material_availability = materials_response.strip()

    #save above parameters in a dictionary
    geo_metadata = {
    "terrain": terrain,
    "climate": climate,
    "logistics": logistics,
    "safety": safety,
    "soil_type": soil_type,
    "material_availability": material_availability
    }

    return geo_metadata
