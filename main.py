import os
import sys
import zipfile
import rarfile
import shutil
import tempfile
import logging
import pandas as pd
from PIL import Image
import sys
from pdf2image import convert_from_path
import openpyxl
import subprocess
import xlrd
import base64
from groq import Groq
import json

from utils import unzip_all_files, find_first_excel_file, convert_list_values_to_markdown, clean_and_format_markdown_with_deepseek
from site_images import process_and_display_images
from location_insights import find_location_parameters
from first_pass import process_files

# Constants
WORKING_DIR = "processed2_files"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "main.log")

# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def cleanup():
    """Remove working directory"""
    try:
        shutil.rmtree(WORKING_DIR)
        logging.info(f"Cleaned up {WORKING_DIR}")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

def ensure_working_directory():
    """Create working directory if not exists"""
    try:
        os.makedirs(WORKING_DIR, exist_ok=True)
        logging.info(f"Created working directory: {WORKING_DIR}")
    except Exception as e:
        logging.error(f"Failed to create working directory: {e}")
        raise

def process(zip_file_path):
    """Main function to process the input zip file"""
    try:
        logging.info(f"Started processing for: {zip_file_path}")
        ensure_working_directory()

        shutil.move(zip_file_path, WORKING_DIR)
        unzip_all_files(WORKING_DIR)
        logging.info("Extraction complete.")

        # Process first pass
        answers = process_files(WORKING_DIR)
        logging.info("Completed first pass document extraction.")

        contract_type = answers.get('tender_type', 'Not found')
        road_length = answers.get('length_of_road', 'Not found')
        road_location = answers.get('road_location', 'Not found')

        if any(val == "Not found" for val in [contract_type, road_length, road_location]):
            logging.warning("Some required fields were not found.")
            return

        geo_location = find_location_parameters(road_location)
        final_images = process_and_display_images(road_location)

        logging.info(f"Geo location: {geo_location}")
        logging.info(f"Images found: {final_images}")

        # Handle contract type-specific logic
        results = {}

        if contract_type in ("EPC", "epc"):
            logging.info("Processing EPC contract type.")
            from epc import analyze_folder, process_zone_bc, process_zone_ab, process_zone_hi, process_zone_cd, extract_zone_bc_image_info

            output = analyze_folder(WORKING_DIR)
            pdf_path = output["Schedule-B"]["pdf"]
            pdf_path2 = output["Schedule-H"]["pdf"]

            # Extract page ranges
            zone_ab_range = (output["Schedule-A"]["page"], output["Schedule-A"]["page"] + 4)
            zone_bc_range = (output["Schedule-B"]["page"], output["Schedule-C"]["page"] - 1)
            zone_cd_range = (output["Schedule-C"]["page"], output["Schedule-D"]["page"] - 1)
            zone_hi_range = (output["Schedule-H"]["page"], output["Schedule-H"]["page"] + 9)

            # Process all zones
            process_zone_ab(pdf_path, *zone_ab_range, results=results)
            process_zone_bc(pdf_path, *zone_bc_range, results=results)
            process_zone_cd(pdf_path, *zone_cd_range, results=results)
            process_zone_hi(pdf_path2, *zone_hi_range, results=results)
            extract_zone_bc_image_info(pdf_path, *zone_bc_range, results=results)
          
            convert_list_values_to_markdown(results)
            clean_and_format_markdown_with_deepseek(results)

        elif contract_type in ("HAM", "ham"):
            logging.info("Processing HAM contract type.")
            from ham import analyze_folder, process_zone_bc, process_zone_ab, process_zone_cd, extract_zone_bc_image_info

            output = analyze_folder(WORKING_DIR)
            pdf_path = output["Schedule-B"]["pdf"]

            zone_ab_range = (output["Schedule-A"]["page"], output["Schedule-A"]["page"] + 4)
            zone_bc_range = (output["Schedule-B"]["page"], output["Schedule-C"]["page"] - 1)
            zone_cd_range = (output["Schedule-C"]["page"], output["Schedule-D"]["page"] - 1)

            process_zone_ab(pdf_path, *zone_ab_range, results=results)
            process_zone_bc(pdf_path, *zone_bc_range, results=results)
            process_zone_cd(pdf_path, *zone_cd_range, results=results)
            extract_zone_bc_image_info(pdf_path, *zone_bc_range, results=results)
            
            convert_list_values_to_markdown(results)
            clean_and_format_markdown_with_deepseek(results)

        elif contract_type in ("ITEM-RATE", "item-rate"):
            logging.info("Processing Item-rate contract type.")
            from item_rate import classify_and_summarize, generate_markdown_summaries, extract_boq_with_deepseek

            file_path = find_first_excel_file(WORKING_DIR)
            final_df = extract_boq_with_deepseek(file_path)
            final_df.to_csv("cleaned_boq_outputs.csv", index=False)

            categories = classify_and_summarize("cleaned_boq_outputs.csv")
            markdown_structure, markdown_road_works, markdown_roadside_furniture = generate_markdown_summaries(categories)

            results.update({
                "categories": categories,
                "markdown_structure": markdown_structure,
                "markdown_road_works": markdown_road_works,
                "markdown_roadside_furniture": markdown_roadside_furniture
            })

        elif contract_type in ("BOT", "bot"):
            logging.info("Processing BOT contract type.")
            # Implementation can be added here

        else:
            logging.error(f"Unknown contract type: {contract_type}")
            return
            
        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.exception(f"Error during processing: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Usage: python main.py <zip_file_path>")
        sys.exit(1)

    zip_file_path = sys.argv[1]
    if not os.path.exists(zip_file_path):
        logging.error(f"File {zip_file_path} does not exist.")
        sys.exit(1)

    logging.info(f"Starting processing for {zip_file_path}")
    process(zip_file_path)

