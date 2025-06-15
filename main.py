import os
import sys
import zipfile
import rarfile
import tempfile
from PIL import Image
import sys
from pdf2image import convert_from_path
import openpyxl
import subprocess
import xlrd
import base64
from groq import Groq
import shutil
import json

import pandas as pd

from utils import unzip_all_files, find_first_excel_file, convert_list_values_to_markdown, clean_and_format_markdown_with_deepseek
from site_images import process_and_display_images
from location_insights import find_location_parameters
from first_pass import process_files

# start a logging session and log file to be saved in logs folder
import logging
logging.basicConfig(
    filename='logs/main.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Working directory
WORKING_DIR = "processed2_files"


def cleanup():
    """Remove working directory"""
    try:
        shutil.rmtree(WORKING_DIR)
        print(f"Cleaned up {WORKING_DIR}")
    except Exception as e:
        print(f"Error during cleanup: {e}")



def process(zip_file_path):
    """Main function to process the input zip file"""
    # print(f"Creating directory: {WORKING_DIR}")
    # Create log
    logging.info(f"Creating directory: {WORKING_DIR}")
    os.makedirs(WORKING_DIR, exist_ok=True)
    # print("Directory creation attempted.")
    logging.info("Directory creation attempted.")

    
    # try:
    # Extract files
    # with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    #     zip_ref.extractall(WORKING_DIR)

    # MOVE THE ZIP FILE TO WORKING DIRECTORY
    # shutil.move(zip_file_path, WORKING_DIR)
    # logging.info(f"Moved {zip_file_path} to {WORKING_DIR}")
    # # # Process nested archives
    # unzip_all_files(WORKING_DIR)
    # logging.info(f"Unzipped all files in {WORKING_DIR}")
    # exit(1)
    # Process documents
    answers = process_files(WORKING_DIR)

    exit(1)
    
    contract_type = answers.get('organization_type', 'Not found').upper()
    road_length = answers.get('length_of_road', 'Not found').upper()
    road_location = answers.get('road_location', 'Not found').upper()

    if any([road_length, contract_type, road_location]) == 'Not found':
        logging.warning("Some answers were not found in the documents.")
        exit(1)
    # # Print final answers
    # print("\n=== Final Extracted Information ===")
    # print(f"Road Length: {answers.get('length_of_road', 'Not found')} km")
    # print(f"Contract Type: {answers.get('organization_type', 'Not found').upper()}")
    # print(f"Location: {answers.get('road_location', 'Not found')}")

    # return answers
    logging.info(f"Processed {zip_file_path} successfully.")



    geo_location = find_location_parameters(road_location)
    logging.info(f"Geo location for {road_location}: {geo_location}")

    final_images = process_and_display_images(road_location)
    logging.info(f"Final images for {road_location}: {final_images}")
    #########
    # Save the final images and geo location to mongodb
    #########
    print(f"Final images: {final_images}")
    print(f"Geo location: {geo_location}")

    

    if contract_type == "epc" or contract_type == "EPC":
        logging.info("Processing for EPC contract type.")
        from epc import analyze_folder, process_zone_bc, process_zone_ab, process_zone_hi, process_zone_cd, extract_zone_bc_image_info

        output = analyze_folder(WORKING_DIR)
        zone_bc_start_page = output.get("Schedule-B").get("page")
        zone_bc_end_page = output.get("Schedule-C").get("page")-1
        zone_cd_start_page = output.get("Schedule-C").get("page")
        zone_cd_end_page = output.get("Schedule-D").get("page")-1
        zone_ab_start_page = output.get("Schedule-A").get("page")
        zone_ab_end_page = output.get("Schedule-A").get("page")+4
        zone_hi_start_page = output.get("Schedule-H").get("page")
        zone_hi_end_page = output.get("Schedule-H").get("page")+9
        pdf_path = output.get("Schedule-B").get("pdf")
        pdf_path2 = output.get("Schedule-H").get("pdf")

        results = {}
        process_zone_bc(pdf_path, zone_bc_start_page, zone_bc_end_page, results=results)
        process_zone_ab(pdf_path, zone_ab_start_page, zone_ab_end_page, results=results)
        process_zone_hi(pdf_path2, zone_hi_start_page, zone_hi_end_page, results=results)
        process_zone_cd(pdf_path, zone_cd_start_page, zone_cd_end_page, results=results)
        extract_zone_bc_image_info(pdf_path, zone_bc_start_page, zone_bc_end_page, results=results)
        convert_list_values_to_markdown(results)
        clean_and_format_markdown_with_deepseek(results)


    elif contract_type == "ham" or contract_type == "HAM":
        logging.info("Processing for HAM contract type.")
        from ham import analyze_folder, process_zone_bc, process_zone_ab, process_zone_cd, extract_zone_bc_image_info

        
        output = analyze_folder(WORKING_DIR)
        zone_bc_start_page = output.get("Schedule-B").get("page")
        zone_bc_end_page = output.get("Schedule-C").get("page")-1
        zone_cd_start_page = output.get("Schedule-C").get("page")
        zone_cd_end_page = output.get("Schedule-D").get("page")-1
        zone_ab_start_page = output.get("Schedule-A").get("page")
        zone_ab_end_page = output.get("Schedule-A").get("page")+4
        pdf_path = output.get("Schedule-B").get("pdf")

        results = {}
        process_zone_bc(pdf_path, zone_bc_start_page, zone_bc_end_page, results=results)
        process_zone_ab(pdf_path, zone_ab_start_page, zone_ab_end_page, results=results)
        process_zone_cd(pdf_path, zone_cd_start_page, zone_cd_end_page, results=results)
        extract_zone_bc_image_info(pdf_path, zone_bc_start_page, zone_bc_end_page, results=results)
        convert_list_values_to_markdown(results)
        clean_and_format_markdown_with_deepseek(results)

        #######
        #send the result to mongo db
        #######

    elif contract_type == "item-rate" or contract_type == "ITEM-RATE":
        logging.info("Processing for Item-rate contract type.")
        folder_path = WORKING_DIR
        from item_rate import classify_and_summarize, generate_markdown_summaries, extract_boq_with_deepseek
        
        file_path = find_first_excel_file(folder_path)
        final_df = extract_boq_with_deepseek(file_path)
        # print(final_df.head())
        final_df.to_csv("cleaned_boq_outputs.csv", index=False)
        csv_path = "cleaned_boq_outputs.csv"  # Update path as needed
        categories = classify_and_summarize(csv_path)

        markdown_structure, markdown_road_works, markdown_roadside_furniture = generate_markdown_summaries(categories)

        # Combine all classified items into one DataFrame
        # combined_boq_df = pd.DataFrame(
        #     categories["Road works (including pavement)"] +
        #     categories["Roadside furniture"] +
        #     categories["Structures Work"]
        # )
        ######################
        # send categories, markdown_structure, markdown_road_works, markdown_roadside_furniture to mangodb
        ######################

        print("Categories:", categories
            )
        print("Markdown Structure:", markdown_structure)
        print("Markdown Road Works:", markdown_road_works)
        print("Markdown Roadside Furniture:", markdown_roadside_furniture)

    elif contract_type == "bot" or contract_type == "BOT":
        logging.info("Processing for BOT contract type.")
        pass

    else:
        logging.error(f"Unknown contract type: {contract_type}")
        exit(1)

        

    # except Exception as e:
    #     # print(f"Error: {e}")
    #     logging.error(f"Error processing {zip_file_path}: {e}")
        # return {}
    # finally:
    #     cleanup()


if __name__ == "__main__":

    zip_file_path = sys.argv[1]
    # zip_file_path = "C:\\Users\\kshub\\OneDrive\\Documents\\KnowledgeEdgeAI\\tender\\folder_org\\downloads copy\\downloads copy\\86860221.zip"

    if not os.path.exists(zip_file_path):
        print(f"File {zip_file_path} does not exist.")
        sys.exit(1)

    logging.info(f"Starting processing for {zip_file_path}")
    process(zip_file_path)
    logging.info("Processing completed.")
