import os
import zipfile
import rarfile
import requests
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
import logging

client = Groq(api_key="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")

TENDER_TYPES = ["item-rate", "epc", "ham", "bot"]

def validate_tender_type(tender_type):
    tender_type = tender_type.lower().strip()
    for valid_type in TENDER_TYPES:
        if valid_type in tender_type:
            return valid_type
    return None

def analyze_with_groq(file_path, answers):
    try:
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:image/png;base64,{encoded_image}"

                completion = client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract these exact details from the tender document:\n"
                                        "1. the Length of road (KM) to be worked on - only numbers like '5.2'\n"
                                        "2. Type of Tender - must be one of: item-rate, epc, ham, bot\n"
                                        "3. Road location name - only official name from document\n"
                                        "Return ONLY as valid JSON with these keys: "
                                        "'length_of_road', 'tender_type', 'road_location'"
                            },
                            {
                                "type": "image_url",
                                "image_url": { "url": image_url }
                            }
                        ]
                    }],
                    temperature=0.1,
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )

        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r') as f:
                text_content = f.read()

                completion = client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=[{
                        "role": "user",
                        "content": f"Extract these exact details from the tender document:\n"
                                   "1. Length of road (KM) to be worked on - only numbers like '5.2'\n"
                                   "2. Type of Tender - must be one of: item-rate, epc, ham, bot\n"
                                   "3. Road location name - only official name from document\n"
                                   "Document content:\n{text_content}\n"
                                   "Return ONLY as valid JSON with these keys: "
                                   "'length_of_road', 'tender_type', 'road_location'"
                    }],
                    temperature=0.1,
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )

        response = completion.choices[0].message.content
        logging.info(f"Analysis for {file_path}: {response}")

        try:
            data = json.loads(response)
            updated = False

            if "length_of_road" in data and not answers.get("length_of_road"):
                try:
                    if data["length_of_road"] is not None:
                        answers["length_of_road"] = float(data["length_of_road"])
                        updated = True
                except (ValueError, TypeError):
                    logging.warning("Invalid value for length_of_road")

            if "tender_type" in data and not answers.get("tender_type"):
                tender_type = validate_tender_type(data["tender_type"])
                if tender_type:
                    answers["tender_type"] = tender_type
                    updated = True

            if "road_location" in data and not answers.get("road_location"):
                answers["road_location"] = data["road_location"].strip()
                updated = True

            if updated:
                logging.info(f"Updated answers: {answers}")

            if all(k in answers for k in ["length_of_road", "tender_type", "road_location"]):
                logging.info("✅ All answers found.")
                return answers

        except json.JSONDecodeError:
            logging.error("Invalid JSON response")

    except Exception as e:
        logging.error(f"Error analyzing with Groq: {e}")

    return answers

def process_pdf(file_path, answers):
    try:
        images = convert_from_path(file_path, first_page=1, last_page=2, poppler_path=r"C:\Program Files\poppler-24.08.0\Library\bin")
        for i, image in enumerate(images):
            screenshot_path = f"{os.path.splitext(file_path)[0]}_page_{i+1}.png"
            image.save(screenshot_path, 'PNG')
            logging.info(f"Converted page {i+1} of PDF to image: {screenshot_path}")
            if analyze_with_groq(screenshot_path, answers):
                return answers
    except Exception as e:
        logging.error(f"Error processing PDF {file_path}: {e}")
    return answers

def process_image(file_path, answers):
    try:
        return analyze_with_groq(file_path, answers)
    except Exception as e:
        logging.error(f"Error processing image {file_path}: {e}")
    return answers

def process_word(file_path, answers):
    try:
        pdf_path = f"{os.path.splitext(file_path)[0]}.pdf"
        result = subprocess.run(["unoconv", "-f", "pdf", "-o", pdf_path, file_path],
                                capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Converted Word to PDF: {pdf_path}")
            return process_pdf(pdf_path, answers)
        else:
            logging.error(f"unoconv failed: {result.stderr}")
    except Exception as e:
        logging.error(f"Error processing Word file {file_path}: {e}")
    return answers

def process_excel(file_path, answers):
    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".xlsx":
            wb = openpyxl.load_workbook(file_path)
            for i, sheetname in enumerate(wb.sheetnames[:2]):
                sheet = wb[sheetname]
                screenshot_path = f"{os.path.splitext(file_path)[0]}_sheet_{i+1}.txt"

                tokens = []
                for row in sheet.iter_rows(values_only=True):
                    row_tokens = ' '.join(map(str, row)).split()
                    tokens.extend(row_tokens)
                    if len(tokens) >= 10000:
                        tokens = tokens[:10000]
                        break

                with open(screenshot_path, 'w') as f:
                    f.write(' '.join(tokens))

                logging.info(f"Saved Excel tokens: {screenshot_path}")
                if analyze_with_groq(screenshot_path, answers):
                    return answers

        elif ext == ".xls":
            wb = xlrd.open_workbook(file_path)
            for i, sheet in enumerate(wb.sheets()[:2]):
                screenshot_path = f"{os.path.splitext(file_path)[0]}_sheet_{i+1}.txt"

                tokens = []
                for row_idx in range(sheet.nrows):
                    row_tokens = ' '.join(map(str, sheet.row_values(row_idx))).split()
                    tokens.extend(row_tokens)
                    if len(tokens) >= 10000:
                        tokens = tokens[:10000]
                        break

                with open(screenshot_path, 'w') as f:
                    f.write(' '.join(tokens))

                logging.info(f"Saved Excel tokens: {screenshot_path}")
                if analyze_with_groq(screenshot_path, answers):
                    return answers

    except Exception as e:
        logging.error(f"Error processing Excel file {file_path}: {e}")
    return answers

def process_files(root_dir):
    answers = {}
    supported_extensions = ('.pdf', '.jpg', '.jpeg', '.doc', '.docx', '.xls', '.xlsx')

    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            if ext in supported_extensions:
                try:
                    logging.info(f"Processing file: {file_path}")
                    if ext == '.pdf' and process_pdf(file_path, answers):
                        return answers
                    elif ext in ('.jpg', '.jpeg') and process_image(file_path, answers):
                        return answers
                    elif ext in ('.doc', '.docx') and process_word(file_path, answers):
                        return answers
                    elif ext in ('.xls', '.xlsx') and process_excel(file_path, answers):
                        return answers
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")

    return answers
