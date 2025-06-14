
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

# Initialize Groq client
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)


# DeepSeek Setup
DEEPSEEK_API_KEY = "sk-fe754eb8e5a04ec79de5c71064b5e25d"  # Replace with your key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

SERPAPI_API_KEY = "ee9869f199c55efdc0ae10df13c2d11b2028c7baf194ef856ab88bd00cf6822a"

"""
Organization Type Enums:
1. Item-rate: Traditional contract where payment is made based on measured quantities of work
2. EPC (Engineering, Procurement, Construction): Turnkey contract where contractor handles all aspects
3. HAM (Hybrid Annuity Model): Public-private partnership with 40% government funding
4. BOT (Build-Operate-Transfer): Private entity builds, operates for concession period, then transfers
"""

TENDER_TYPES = ["item-rate", "epc", "ham", "bot"]


def unzip_all_files(root_dir):
    """Recursively unzip all ZIP and RAR files in directory"""
    extracted_files = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith('.zip'):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        extract_path = os.path.join(root, os.path.splitext(file)[0])
                        os.makedirs(extract_path, exist_ok=True)
                        zip_ref.extractall(extract_path)
                        extracted_files.append(extract_path)
                        extracted_files.extend(unzip_all_files(extract_path))

                elif file.lower().endswith('.rar'):
                    with rarfile.RarFile(file_path, 'r') as rar_ref:
                        extract_path = os.path.join(root, os.path.splitext(file)[0])
                        os.makedirs(extract_path, exist_ok=True)
                        rar_ref.extractall(extract_path)
                        extracted_files.append(extract_path)
                        extracted_files.extend(unzip_all_files(extract_path))

            except (zipfile.BadZipFile, rarfile.BadRarFile) as e:
                print(f"Error extracting {file_path}: {e}")

    return extracted_files

def validate_tender_type(tender_type):
    """Validate and normalize tender type"""
    tender_type = tender_type.lower().strip()
    for valid_type in TENDER_TYPES:
        if valid_type in tender_type:
            return valid_type
    return None

def analyze_with_groq(file_path, answers):
    """Analyze document with Groq's Llama model to extract required info"""
    try:
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:image/png;base64,{encoded_image}"

                completion = client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract these exact details from the tender document:\n"
                                            "1. the Length of road to be worked on - only numbers like '5.2'\n"
                                            "2. Type of Tender - must be one of: item-rate, epc, ham, bot\n"
                                            "3. Road location name - only official name from document\n"
                                            "Return ONLY as valid JSON with these keys: "
                                            "'length_of_road', 'tender_type', 'road_location'"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.1,  # Low temperature for precise answers
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )

        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r') as f:
                text_content = f.read()

                completion = client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Extract these exact details from the tender document:\n"
                                      "1. Length of road to be worked on - only numbers like '5.2'\n"
                                      "2. Type of Tender - must be one of: item-rate, epc, ham, bot\n"
                                      "3. Road location name - only official name from document\n"
                                      "Document content:\n{text_content}\n"
                                      "Return ONLY as valid JSON with these keys: "
                                      "'length_of_road', 'tender_type', 'road_location'"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )

        response = completion.choices[0].message.content
        print(f"Analysis for {file_path}: {response}")

        try:
            data = json.loads(response)
            updated = False

            # Validate and update answers
            if "length_of_road" in data and not answers.get("length_of_road"):
                try:
                    answers["length_of_road"] = float(data["length_of_road"])
                    updated = True
                except ValueError:
                    pass

            if "tender_type" in data and not answers.get("tender_type"):
                tender_type = validate_tender_type(data["tender_type"])
                if tender_type:
                    answers["tender_type"] = tender_type
                    updated = True

            if "road_location" in data and not answers.get("road_location"):
                answers["road_location"] = data["road_location"].strip()
                updated = True

            if updated:
                print(f"Updated answers: {answers}")

            # Check if we have all answers
            if all(k in answers for k in ["length_of_road", "tender_type", "road_location"]):
                print("\n✅ All answers found!")
                return True

        except json.JSONDecodeError:
            print("Invalid JSON response")

    except Exception as e:
        print(f"Error analyzing with Groq: {e}")

    return False

def process_pdf(file_path, answers):
    """Process PDF file - convert first 2 pages to images and analyze"""
    try:
        images = convert_from_path(file_path, first_page=1, last_page=2)
        for i, image in enumerate(images):
            screenshot_path = f"{os.path.splitext(file_path)[0]}_page_{i+1}.png"
            image.save(screenshot_path, 'PNG')
            if analyze_with_groq(screenshot_path, answers):
                return True
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return False

def process_image(file_path, answers):
    """Process image file and analyze"""
    try:
        if analyze_with_groq(file_path, answers):
            return True
    except Exception as e:
        print(f"Error processing image: {e}")
    return False

def process_word(file_path, answers):
    """Convert DOC/DOCX to PDF and process"""
    try:
        pdf_path = f"{os.path.splitext(file_path)[0]}.pdf"
        result = subprocess.run(["unoconv", "-f", "pdf", "-o", pdf_path, file_path],
                              capture_output=True, text=True)

        if result.returncode == 0:
            return process_pdf(pdf_path, answers)
    except Exception as e:
        print(f"Error processing Word file: {e}")
    return False

def process_excel(file_path, answers):
    """Process Excel file and analyze - first 10000 tokens only"""
    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".xlsx":
            wb = openpyxl.load_workbook(file_path)
            for i, sheetname in enumerate(wb.sheetnames[:2]):  # First 2 sheets only
                sheet = wb[sheetname]
                screenshot_path = f"{os.path.splitext(file_path)[0]}_sheet_{i+1}.txt"

                # Extract first 10000 tokens
                tokens = []
                for row in sheet.iter_rows(values_only=True):
                    row_tokens = ' '.join(map(str, row)).split()
                    tokens.extend(row_tokens)
                    if len(tokens) >= 10000:
                        tokens = tokens[:10000]
                        break

                # Save first 10000 tokens
                with open(screenshot_path, 'w') as f:
                    f.write(' '.join(tokens))

                print(f"Saved first 10000 tokens: {screenshot_path}")
                if analyze_with_groq(screenshot_path, answers):
                    return True

        elif ext == ".xls":
            wb = xlrd.open_workbook(file_path)
            for i, sheet in enumerate(wb.sheets()[:2]):  # First 2 sheets only
                screenshot_path = f"{os.path.splitext(file_path)[0]}_sheet_{i+1}.txt"

                # Extract first 10000 tokens
                tokens = []
                for row_idx in range(sheet.nrows):
                    row_tokens = ' '.join(map(str, sheet.row_values(row_idx))).split()
                    tokens.extend(row_tokens)
                    if len(tokens) >= 10000:
                        tokens = tokens[:10000]
                        break

                # Save first 10000 tokens
                with open(screenshot_path, 'w') as f:
                    f.write(' '.join(tokens))

                print(f"Saved first 10000 tokens: {screenshot_path}")
                if analyze_with_groq(screenshot_path, answers):
                    return True

    except Exception as e:
        print(f"Error processing Excel file: {e}")
    return False

def process_files(root_dir):
    """Process all supported files in directory"""
    answers = {}
    supported_extensions = ('.pdf', '.jpg', '.jpeg', '.doc', '.docx', '.xls', '.xlsx')

    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            if ext in supported_extensions:
                try:
                    if ext == '.pdf' and process_pdf(file_path, answers):
                        return answers
                    elif ext in ('.jpg', '.jpeg') and process_image(file_path, answers):
                        return answers
                    elif ext in ('.doc', '.docx') and process_word(file_path, answers):
                        return answers
                    elif ext in ('.xls', '.xlsx') and process_excel(file_path, answers):
                        return answers
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return answers

def extract_page_content(page):
    elements, table_bboxes = [], []
    for table in page.find_tables():
        table_bboxes.append(table.bbox)
        elements.append({"type": "table", "top": float(table.bbox[1]), "content": table.extract()})

    words = page.extract_words()
    grouped_lines = []
    for word in words:
        x0, x1, top, bottom = float(word["x0"]), float(word["x1"]), float(word["top"]), float(word["bottom"])
        if any(x0 >= bx0 and x1 <= bx1 and top >= by0 and bottom <= by1 for (bx0, by0, bx1, by1) in table_bboxes):
            continue
        for line in grouped_lines:
            if abs(line["top"] - top) <= 2:
                line["words"].append((x0, word["text"]))
                break
        else:
            grouped_lines.append({"top": top, "words": [(x0, word["text"])]})

    for line in grouped_lines:
        line["words"].sort()
        elements.append({"type": "text", "top": line["top"], "content": " ".join(word for _, word in line["words"])})
    elements.sort(key=lambda e: e["top"])

    output = []
    for el in elements:
        if el["type"] == "table":
            table_text = "\n".join(" | ".join(cell or "" for cell in row) for row in el["content"])
            output.append(f"Table:\n{table_text}")
        else:
            output.append(f"Text:\n{el['content']}")
    return "\n\n".join(output)


def find_first_excel_file(root_dir):
    """
    Walks through the given directory and its subdirectories to find the first
    .xls or .xlsx file. Returns the full file path if found, else None.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(('.xls', '.xlsx')):
                return os.path.join(dirpath, file)
    return None

def call2_deepseek(system_prompt, user_prompt, MODEL_NAME="deepseek-reasoner"):

    strict_markdown_prompt = """
      *Instruction*:
      You are a geo expert, reason well and provide info, you can guess if you do not have concrete answers. The doc can contain info about multiple projects, but we want only on the project of THIS RFP. Your response must be in *pure GitHub-flavored markdown*, optimized for direct rendering on a UI. Follow these rules absolutely:

      1. *Format Requirements*:
        - Only use:
          - Do NOT use Headings
          - Tables (| Column | Data |)
          - Bullet points (- **Key**: Value)
          - Bold for labels (**Estimated Cost**: ₹X Cr)
          - ### Headings
          - Tables (| Column | Data |)
          - Bullet points (- **Key**: Value)
          - Bold for labels (**Estimated Cost**: ₹X Cr)
          - short Explanation statement MAXIMUM 1-2 line in strict markdown format
        - Never:
          - Add headings
          - Add conversational fluff ("After reviewing...")
        - *Do not enclose the markdown in triple backticks or markdown code fences.*
        """

    combined_prompt = system_prompt + "\n\n" + strict_markdown_prompt

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": combined_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ DeepSeek API error: {response.status_code} - {response.text}"



def query_deepseek(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

    try:
        if response.status_code != 200:
            print("❗ Non-200 response from DeepSeek:")
            print("Status:", response.status_code)
            print("Response:", response.text)
            return ""

        data = response.json()
        if "choices" not in data:
            print("❗ 'choices' missing in response:", json.dumps(data, indent=2))
            return ""

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("❌ Failed to parse DeepSeek response")
        print("Status code:", response.status_code)
        print("Raw response:", response.text)
        raise e
    
def convert_list_values_to_markdown(results: dict) -> None:
    for key in results:
        value = results[key]

        if isinstance(value, list):
            markdown = f"## {key.replace('_', ' ').title()}\n"
            for idx, item in enumerate(value, 1):
                markdown += f"{idx}. {item.strip()}\n"
            results[key] = markdown  # Replace the list with markdown string

    results["GEOMETRIC_DESIGN"] = results["GEOMETRIC_DESIGN"] + results["PAVEMENT"] + results["IMAGE_SUMMARY"]
    results["TRAFFIC_CONTROL"] += results["PROJECT_FACILITIES"]
    del results["PAVEMENT"]
    del results["PROJECT_FACILITIES"]
    del results["IMAGE_SUMMARY"]

def clean_and_format_markdown_with_deepseek(results: dict) -> None:
    for key, markdown_content in results.items():
        if key == "CURRENT_SITE":
            continue
        print(f"\n📘 Processing section: {key}...")

        prompt = f"""You are a civil engineering analyst writing an internal summary.

                  We’ve collected markdown notes on the topic **{key.replace('_', ' ').title()}** from various sources.
                  Please refine and present this information in a clean, readable markdown format.

                  **Format Requirements**:
                              - Only use:
                                - Tables (`| Column | Data |`)
                                - Bullet points (`- **Key**: Value`)
                                - Bold for labels (`**Estimated Cost**: ₹X Cr`)
                              - Never:
                                - Use the ### initial heading
                                - have the content in```markdown
                                - Cite pages/sources ("As per Page 12...")
                                - Add conversational fluff ("After reviewing...")
                                - Explain missing data (use `[Not specified]`)
                              - **Do not enclose the markdown in triple backticks or markdown code fences.**

                  **Priority Order**:
                              - Tables for structured data (costs, schedules)
                              - Bullet lists for descriptive fields (terrain, materials)
                              - Bold labels for key-value pairs

                  For Tables**
                              - Do NOT include items with zero percent weightage
                              - Do NOT leave any cells blank. Every row must explicitly include: Sub-Work, Stage ,Weightage. If a Sub-Work or Stage repeats across multiple rows, repeat it explicitly in each row.
                              - Avoid using ditto marks, hyphens (-), or blank cells to imply repeated values — always fill them in.
                              - Keep a seperate table for every major work, along with their subworks and percentages
                              - Do not forget to include the percentge of major work in heading above table

                  **Examples**:
                              ### Payment Weightage
                              | Work Item        | Stage          | Weightage |
                              |------------------|----------------|-----------|
                              | Road Works       | Earthwork      | 22.70%    |
                              | Protection Works | Breast Wall    | 59.32%    |

                  ### Guidelines:
                  - Do not miss even a single work item which has non zero work/quantity/Dimensions
                  - Remove any clause or index numbers like `2.4`, `4.3.1`, etc.
                  - Combine redundant lines or bullet points.
                  - Make it look like it’s our own structured technical analysis.
                  - Avoid any conversational phrases like “the document says” or “as mentioned above”.
                  - Preserve all meaningful data in a concise and organized format.

                  🚫 **Strictly DO NOT include** items that:

                  - Are labeled or described as "NIL", "Zero", "Not specified", "not available", "not mentioned", "TBD", "indicative only", or similar.
                  - Have **no dimensions, quantity, or % weightage** mentioned.
                  - Have **zero weightage** or are marked as having "nil %"

                  Here is the raw content:

                  {markdown_content}
                  """

        cleaned_markdown = query_deepseek(prompt)
        print(cleaned_markdown)
        results[key] = cleaned_markdown
