import pandas as pd
import json
import requests
import re
import logging

from utils import query_deepseek

# ===== Prompt Builder =====
def build_prompt_using_list_format(df):
    sample = df.iloc[:15].values.tolist()

    prompt = f"""
You are an expert in Indian Government BOQ (Bill of Quantities) data extraction from Excel sheets.

The sheet is messy and may contain:
- Multiline descriptions
- Merged cells
- Misaligned headers

Below are first 15 rows from such an Excel sheet, formatted as lists of cell values. Each sublist represents a row, and each index represents a column (0-indexed). Your job is to:

1. Identify which row index (out of the 3 rows shown) contains the actual column headers (like 'Sl. No.', 'Item Description', 'Quantity', 'Unit', etc.).
2. Determine which column index (0-based) corresponds to each of the following:
    - serial_number
    - item_description
    - quantity
    - units
    - estimated_rate
    - total_amount

Return only this JSON (strict format, no extra explanation):

{{
  "header_row": <int>,  // must be 0, 1 or 2
  "columns": {{
    "serial_number": <int>,
    "item_description": <int>,
    "quantity": <int>,
    "units": <int>,
    "estimated_rate": <int>,
    "total_amount": <int>
  }}
}}

Here are the rows:
{json.dumps(sample, indent=2)}
"""
    return prompt

def clean_llm_json_response(response_text):
    return re.sub(r"^```(?:json|markdown)?|```$", "", response_text.strip(), flags=re.MULTILINE)

# ===== Extraction Pipeline =====
def extract_boq_with_deepseek(file_path):
    logger.info(f"🔍 Loading Excel file: {file_path}")
    df = pd.read_excel(file_path, sheet_name=0, header=None)

    df.dropna(axis=0, how='all', inplace=True)
    df = df.dropna(axis=1, thresh=15)

    logger.info("🧠 Building DeepSeek prompt...")
    prompt = build_prompt_using_list_format(df)

    logger.info("📤 Sending prompt to DeepSeek...")
    llm_response = query_deepseek(prompt)
    cleaned_response = clean_llm_json_response(llm_response)
    logger.debug(f"🧾 Cleaned LLM response: {cleaned_response}")

    try:
        structure = json.loads(cleaned_response)
    except Exception as e:
        logger.error(f"❌ Failed to parse DeepSeek response: {e}")
        raise

    header_row = structure["header_row"]
    columns = structure["columns"]

    logger.info("📥 Reading Excel again with correct header...")
    df = pd.read_excel(file_path, sheet_name=0, header=header_row)
    df = df.rename(columns=lambda x: str(x).strip())

    extracted_df = pd.DataFrame({
        "Serial Number": df.iloc[:, columns["serial_number"]],
        "Item Description": df.iloc[:, columns["item_description"]],
        "Quantity": df.iloc[:, columns["quantity"]],
        "Units": df.iloc[:, columns["units"]],
        "Estimated Rate": df.iloc[:, columns["estimated_rate"]],
        "Total Amount (Rs.)": df.iloc[:, columns["total_amount"]],
    })

    if len(extracted_df) > 20:
        logger.info("✂️ Trimming to first 20 rows")
        extracted_df = extracted_df.iloc[:20]

    logger.info("✅ BOQ Extraction Complete")
    return extracted_df

# ===== Classification + Summarization =====
def classify_and_summarize(csv_path):
    logger.info(f"📄 Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    str_cols = df.select_dtypes(include=["object"]).columns
    df[str_cols] = df[str_cols].fillna("")
    all_rows = df.to_dict(orient="records")
    chunk_size = 10

    category_tables = {
        "Road works (including pavement)": [],
        "Roadside furniture": [],
        "Structures Work": []
    }

    for i in range(0, len(all_rows), chunk_size):
        chunk = all_rows[i:i + chunk_size]
        logger.info(f"🔍 Classifying chunk {i // chunk_size + 1}")

        prompt = (
            f"You are given a set of BOQ items. Classify **each row** strictly into one of these 3 categories:\n\n"
            f"1. Road works (including pavement)\n"
            f"2. Roadside furniture\n"
            f"3. Structures Work\n\n"
            f"Return in JSON like this:\n"
            f"{{\n"
            f"  \"Road works (including pavement)\": [rows...],\n"
            f"  \"Roadside furniture\": [rows...],\n"
            f"  \"Structures Work\": [rows...]\n"
            f"}}\n\n"
            f"Each row is like this:\n{json.dumps(chunk, indent=2)}\n"
            f"Only return the json and nothing else, no explanations, in terms of empty cell, leave them empty"
        )

        response = query_deepseek(prompt)

        if not response.strip():
            logger.warning("⚠️ Empty or invalid LLM response")
            continue

        cleaned = clean_llm_json_response(response)
        logger.debug(f"🔍 Cleaned JSON: {cleaned}")

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON Decode Error: {e}")
            raise

        for key in category_tables:
            category_tables[key].extend(result.get(key, []))

    logger.info("✅ Classification complete")
    return category_tables

# ===== Markdown Summaries =====
def generate_markdown_summaries(category_tables):
    def build_prompt(title, items):
        return (
            f"Analyze the following BOQ items under the category **{title}**:\n\n"
            f"```\n{json.dumps(items, indent=2)}\n```\n\n"
            f"Generate:\n"
            f"-If category is empty just mention 'No work in this category' only"
            f"- Table insights strictly in **Markdown** format, keep every row in the table, dont miss any row even if empty, but summarize, add a description column describing the item if necessary, no need to show clauses and repetitive data.\n"
            f"- Use **headings**, **subpoints**, and **bullet points**, only give table insights and key takeaways and NOTHING ELSE (NOT EVEN BOQ ANALYSIS HEADING), keep key takeaways short with not more than 100 tokens, no need for anything else."
        )

    prompts = {
        "Structures Work": build_prompt("Structures Work", category_tables["Structures Work"]),
        "Road works (including pavement)": build_prompt("Road works (including pavement)", category_tables["Road works (including pavement)"]),
        "Roadside furniture": build_prompt("Roadside furniture", category_tables["Roadside furniture"])
    }

    markdown_outputs = {}

    for key, prompt in prompts.items():
        logger.info(f"📝 Generating markdown for: {key}")
        markdown = query_deepseek(prompt)
        markdown_outputs[key] = markdown
        logger.debug(f"📄 Markdown output for {key}:\n{markdown}")

    return (
        markdown_outputs["Structures Work"],
        markdown_outputs["Road works (including pavement)"],
        markdown_outputs["Roadside furniture"]
    )
