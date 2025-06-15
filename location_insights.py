import logging
import requests
from utils import call2_deepseek

def find_location_parameters(road_location):
    logging.info(f"Starting geographic analysis for: {road_location}")

    # 1. Terrain
    try:
        terrain_response = call2_deepseek(
            "You are a geographical analyst. Provide the terrain type in 1-2 lines.",
            f"What is the terrain type at {road_location}? Respond with the terrain type."
        )
        terrain = terrain_response.strip()
        logging.info(f"Terrain for {road_location}: {terrain}")
    except Exception as e:
        logging.error(f"Error fetching terrain info: {e}")
        terrain = "Unknown"

    # 2. Climate and Working Season
    try:
        climate_response = call2_deepseek(
            "You are a geographical analyst. Provide the climate zone and the working season for road construction in 1-2 line.",
            f"What is the climate zone and suitable working season at {road_location}? Respond with the climate type and working season."
        )
        climate = climate_response.strip()
        logging.info(f"Climate and working season for {road_location}: {climate}")
    except Exception as e:
        logging.error(f"Error fetching climate info: {e}")
        climate = "Unknown"

    # 3. Logistical Difficulty
    try:
        logistics_response = call2_deepseek(
            "You are a geographical analyst. Provide 1-2 lines on logistical difficulty.",
            f"What is the logistical difficulty for construction at {road_location}? Respond with the difficulty."
        )
        logistics = logistics_response.strip()
        logging.info(f"Logistics for {road_location}: {logistics}")
    except Exception as e:
        logging.error(f"Error fetching logistics info: {e}")
        logistics = "Unknown"

    # 4. Safety and Threats
    try:
        safety_response = call2_deepseek(
            "You are a geographical analyst. Provide the human safety/terrorism critique in 1-2 lines.",
            f"What is the human threat/terrorism risk at {road_location}? Respond with the risk level."
        )
        safety = safety_response.strip()
        logging.info(f"Safety risk for {road_location}: {safety}")
    except Exception as e:
        logging.error(f"Error fetching safety info: {e}")
        safety = "Unknown"

    # 5. Soil Type and Rock Availability
    try:
        soil_response = call2_deepseek(
            "You are a geographical analyst. Provide the soil type and rock availability in 1-2 line.",
            f"What is the soil type and are rocks available for aggregates at {road_location}? Respond with this information."
        )
        soil_type = soil_response.strip()
        logging.info(f"Soil and rock info for {road_location}: {soil_type}")
    except Exception as e:
        logging.error(f"Error fetching soil info: {e}")
        soil_type = "Unknown"

    # 6. Material Availability
    try:
        materials_response = call2_deepseek(
            "You are a geographical analyst. Provide fuel/cement vendor availability nearby in 1-2 line.",
            f"Are diesel/petrol pumps and cement vendors available near {road_location}? Respond with this availability info."
        )
        material_availability = materials_response.strip()
        logging.info(f"Material availability for {road_location}: {material_availability}")
    except Exception as e:
        logging.error(f"Error fetching material availability info: {e}")
        material_availability = "Unknown"

    geo_metadata = {
        "terrain": terrain,
        "climate": climate,
        "logistics": logistics,
        "safety": safety,
        "soil_type": soil_type,
        "material_availability": material_availability
    }

    logging.info(f"Completed geo analysis for {road_location}: {geo_metadata}")
    return geo_metadata
