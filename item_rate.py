import pandas as pd
import json
import requests
import re

from utils import query_deepseek

# Step 1: Build prompt for DeepSeek
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
    """
    Removes Markdown code formatting from LLM output
    """
    # Remove triple backticks and language label (like ```json)
    return re.sub(r"^```(?:json|markdown)?|```$", "", response_text.strip(), flags=re.MULTILINE)


# Step 3: Extraction Pipeline
def extract_boq_with_deepseek(file_path):
    df = pd.read_excel(file_path, sheet_name=0, header=None)

    # Pre-cleaning
    df.dropna(axis=0, how='all', inplace=True)
    df = df.dropna(axis=1, thresh=15)

    prompt = build_prompt_using_list_format(df)

    llm_response = query_deepseek(prompt)
    cleaned_response = clean_llm_json_response(llm_response)  # Strip markdown
    print(cleaned_response)
    structure = json.loads(cleaned_response)

    header_row = structure["header_row"]
    columns = structure["columns"]

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
        extracted_df = extracted_df.iloc[:20]

    return extracted_df




# ========== Classification + Summarization ==========
def classify_and_summarize(csv_path):
    df = pd.read_csv(csv_path)

    # Fill NaNs for object/string columns
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
            print("❌ Empty or invalid response from LLM for chunk:")
            print(json.dumps(chunk, indent=2))
            continue

        cleaned = clean_llm_json_response(response)
        print(f"\n===== Chunk {i // chunk_size} =====\n{cleaned}\n")

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("❌ Failed to decode JSON from cleaned response:")
            print(cleaned)
            raise e

        for key in category_tables:
            category_tables[key].extend(result.get(key, []))

    return category_tables


# ========== Markdown Summaries ==========
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

    prompt_structure = build_prompt("Structures Work", category_tables["Structures Work"])
    prompt_road = build_prompt("Road works (including pavement)", category_tables["Road works (including pavement)"])
    prompt_furniture = build_prompt("Roadside furniture", category_tables["Roadside furniture"])

    markdown_structure = query_deepseek(prompt_structure)
    markdown_road_works = query_deepseek(prompt_road)
    markdown_roadside_furniture = query_deepseek(prompt_furniture)

    print(f"\n===== Structures Work =====\n{markdown_structure}\n")
    print(f"\n===== Road Works =====\n{markdown_road_works}\n")
    print(f"\n===== Roadside Furniture =====\n{markdown_roadside_furniture}\n")

    return markdown_structure, markdown_road_works, markdown_roadside_furniture






# def clean_llm_json_response(response_text):
#     return re.sub(r"^```(?:json|markdown)?|```$", "", response_text.strip(), flags=re.MULTILINE)

# # ========== Classification from PDF ==========
# def classify_pdf_boq(pdf_path, start_page=1, end_page=30, page_chunk_size=2):
#     category_tables = {
#         "Road works (including pavement)": [],
#         "Roadside furniture": [],
#         "Structures Work": []
#     }

#     with pdfplumber.open(pdf_path) as pdf:
#         total_pages = min(end_page, len(pdf.pages))

#         for i in range(start_page - 1, total_pages, page_chunk_size):
#             pages = pdf.pages[i:i + page_chunk_size]
#             raw_text = "\n".join([p.extract_text() for p in pages if p.extract_text()])

#             if not raw_text.strip():
#                 continue

#             prompt = (
#                 f"You are given a set of BOQ items extracted from a PDF. Classify **each row** strictly into one of these 3 categories:\n\n"
#                 f"1. Road works (including pavement)\n"
#                 f"2. Roadside furniture\n"
#                 f"3. Structures Work\n\n"
#                 f"Return in JSON like this:\n"
#                 f"{{\n"
#                 f"  \"Road works (including pavement)\": [rows...],\n"
#                 f"  \"Roadside furniture\": [rows...],\n"
#                 f"  \"Structures Work\": [rows...]\n"
#                 f"}}\n\n"
#                 f"The data is below (treat each row separately):\n\n"
#                 f"```\n{raw_text}\n```\n"
#                 f"Only return the JSON, do not explain. If a cell is empty, leave it empty."
#             )

#             response = query_deepseek(prompt)
#             if not response.strip():
#                 print("❌ Empty or invalid response from LLM for PDF chunk.")
#                 continue

#             cleaned = clean_llm_json_response(response)
#             print(f"\n===== Chunk pages {i+1}–{i + len(pages)} =====\n{cleaned}\n")

#             try:
#                 result = json.loads(cleaned)
#                 for key in category_tables:
#                     category_tables[key].extend(result.get(key, []))
#             except json.JSONDecodeError as e:
#                 print("❌ Failed to decode JSON from cleaned response.")
#                 print(cleaned)
#                 raise e

#     return category_tables


# # ========== Markdown Summaries ==========
# def generate_markdown_summaries(category_tables):
#     def build_prompt(title, items):
#         return (
#             f"Analyze the following BOQ items under the category **{title}**:\n\n"
#             f"```\n{json.dumps(items, indent=2)}\n```\n\n"
#             f"Generate:\n"
#             f"-If category is empty just mention 'No work in this category' only"
#             f"- Table insights strictly in **Markdown** format, keep every row in the table, don't miss any row even if empty, but summarize, add a description column describing the item if necessary, no need to show clauses and repetitive data.\n"
#             f"- Use **headings**, **subpoints**, and **bullet points**, only give table insights and key takeaways and NOTHING ELSE (NOT EVEN BOQ ANALYSIS HEADING), keep key takeaways short with not more than 100 tokens, no need for anything else."
#         )

#     prompt_structure = build_prompt("Structures Work", category_tables["Structures Work"])
#     prompt_road = build_prompt("Road works (including pavement)", category_tables["Road works (including pavement)"])
#     prompt_furniture = build_prompt("Roadside furniture", category_tables["Roadside furniture"])

#     markdown_structure = query_deepseek(prompt_structure)
#     markdown_road_works = query_deepseek(prompt_road)
#     markdown_roadside_furniture = query_deepseek(prompt_furniture)

#     print(f"\n===== Structures Work =====\n{markdown_structure}\n")
#     print(f"\n===== Road Works =====\n{markdown_road_works}\n")
#     print(f"\n===== Roadside Furniture =====\n{markdown_roadside_furniture}\n")

#     return markdown_structure, markdown_road_works, markdown_roadside_furniture


# # ========== Main ==========
# if __name__ == "__main__":
#     pdf_path = "boq_extract.pdf"  # 🔁 Update this path to your input PDF
#     categories = classify_pdf_boq(pdf_path)

#     markdown_structure, markdown_road_works, markdown_roadside_furniture = generate_markdown_summaries(categories)

#     # Combine all classified items into one DataFrame
#     combined_boq_df = pd.DataFrame(
#         categories["Road works (including pavement)"] +
#         categories["Roadside furniture"] +
#         categories["Structures Work"]
#     )

#     # Display first few rows
#     print("\n===== Combined BOQ Table =====")
#     print(combined_boq_df.head(200).to_string(index=False, na_rep=''))
