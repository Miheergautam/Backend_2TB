import pdfplumber
import requests
from enum import Enum
from typing import Dict, List
import logging
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os
import fitz
from PIL import Image
import io
import base64
from groq import Groq

# Initialize Groq client
groq_client = Groq(api_key="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")

from Backend_2TB.utils import query_deepseek
from Backend_2TB.utils import extract_page_content


# Set up logging  
logging.basicConfig(level=logging.INFO)


# =============================================================================
# Retriever Functions
# =============================================================================


def normalize(text):
    return re.sub(r'[\s\-]+', '', text.lower())

def extract_chunks_with_metadata(pdf_path, chunk_size=400, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if not text:
            continue
        splits = splitter.split_text(text)
        for chunk in splits:
            all_chunks.append(Document(page_content=chunk, metadata={"page": page_num}))
    return all_chunks

def filter_chunks_by_keyword(docs, keywords, combinations=None):
    if isinstance(keywords, str):
        keywords = [keywords]
    norm_keywords = [normalize(k) for k in keywords]

    if combinations:
        for idx, combo in enumerate(combinations):
            selected_keywords = [norm_keywords[i - 1] for i in combo]
            filtered = [
                doc for doc in docs
                if all(k in normalize(doc.page_content) for k in selected_keywords)
            ]
            if filtered:
                return filtered, idx
    else:
        filtered = [
            doc for doc in docs
            if all(k in normalize(doc.page_content) for k in norm_keywords)
        ]
        if filtered:
            return filtered, 0

    return [], (len(combinations) if combinations else 1)

def retrieve_by_keywords_only(all_chunks, keywords, combinations):
    results, combo_index = filter_chunks_by_keyword(all_chunks, keywords, combinations)

    if results:
        top = results[0]
        return top.metadata["page"] + 1, top.page_content, combo_index

    return None, "", (len(combinations) if combinations else 1)

def analyze_folder(folder_path):
    final_results = {}
    schedule_page_index = {
        "Schedule-A": {},
        "Schedule-B": {},
        "Schedule-C": {},
        "Schedule-D": {},
        "Schedule-H": {},
        "Schedule-I": {}
    }

    targets = {
        "Schedule-A": {"keywords": ["Schedule-A", "this Schedule-A"], "combinations": [(2, 1)]},
        "Schedule-B": {"keywords": ["Schedule-B", "this Schedule-B"], "combinations": [(2, 1)]},
        "Schedule-C": {"keywords": ["Project Facilities", "provisions of this agreement"], "combinations": [(1, 2)]},
        "Schedule-D": {"keywords": ["Schedule-D", "this Schedule-D"], "combinations": [(2,), (1,)]},
        "Schedule-H": {"keywords": ["the contract price for this agreement", "Stage"], "combinations": [(1, 2), (1,)]},
        "Schedule-I": {"keywords": ["this Schedule-I", "Annex-I"], "combinations": [(1, 2), (1,)]},
        "Notice Inviting Bid": {"keywords": ["brief particulars of the project"], "combinations": [(1,)]}
    }

    abcd_candidates = []
    hi_candidates = []

    for root, dirs, files in os.walk(folder_path):
        print(f"\n📂 Scanning directory: {root}")
        for filename in files:
            if not filename.lower().endswith(".pdf") or filename.startswith("._"):
                print(f"⏭️ Skipping file: {filename}")
                continue

            full_path = os.path.join(root, filename)
            print(f"\n📄 Processing file: {filename}")

            try:
                all_chunks = extract_chunks_with_metadata(full_path)
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")
                continue

            # Check Schedule A-D
            abcd_found = {}
            combo_score = 0
            all_found = True

            for sched in ["Schedule-A", "Schedule-B", "Schedule-C", "Schedule-D"]:
                page, text, combo_idx = retrieve_by_keywords_only(
                    all_chunks,
                    targets[sched]["keywords"],
                    targets[sched]["combinations"]
                )
                if page:
                    abcd_found[sched] = {"page": page, "pdf": full_path, "combo_index": combo_idx}
                    combo_score += combo_idx
                else:
                    print(f"❌ {sched} not found in {filename}")
                    all_found = False
                    break

            if all_found:
                pages = [abcd_found[s]["page"] for s in ["Schedule-A", "Schedule-B", "Schedule-C", "Schedule-D"]]
                if sorted(pages) == pages and len(set(v["pdf"] for v in abcd_found.values())) == 1:
                    abcd_candidates.append({"set": abcd_found, "combo_score": combo_score})
                    print(f"✅ Found A–D in order in {filename} | Combo Score: {combo_score}")
                else:
                    print(f"⚠️ A–D found, but not in order | Combo Score: {combo_score}")

            # Check Schedule H-I
            hi_found = {}
            hi_combo_score = 0
            hi_all_found = True

            for sched in ["Schedule-H", "Schedule-I"]:
                page, text, combo_idx = retrieve_by_keywords_only(
                    all_chunks,
                    targets[sched]["keywords"],
                    targets[sched]["combinations"]
                )
                if page:
                    hi_found[sched] = {"page": page, "pdf": full_path, "combo_index": combo_idx}
                    hi_combo_score += combo_idx
                else:
                    print(f"❌ {sched} not found in {filename}")
                    hi_all_found = False
                    break

            if hi_all_found:
                h_page = hi_found["Schedule-H"]["page"]
                i_page = hi_found["Schedule-I"]["page"]
                if h_page < i_page and hi_found["Schedule-H"]["pdf"] == hi_found["Schedule-I"]["pdf"]:
                    hi_candidates.append({"set": hi_found, "combo_score": hi_combo_score})
                    print(f"✅ Found H–I in order in {filename} | Combo Score: {combo_score}")
                else:
                    print(f"⚠️ H–I found, but not in order | Combo Score: {combo_score}")

            # Check NIB
            if "Notice Inviting Bid" not in final_results:
                page, text, combo_idx = retrieve_by_keywords_only(
                    all_chunks,
                    targets["Notice Inviting Bid"]["keywords"],
                    targets["Notice Inviting Bid"]["combinations"]
                )
                if page:
                    final_results["Notice Inviting Bid"] = {
                        "page": page,
                        "pdf": full_path
                    }

    if not abcd_candidates:
        raise ValueError("❌ Schedule A–D could not be found in any PDF. Aborting.")

    if not hi_candidates:
        raise ValueError("❌ Schedule H–I could not be found in any PDF. Aborting.")

    if abcd_candidates:
        best = min(abcd_candidates, key=lambda x: x["combo_score"])
        for k, v in best["set"].items():
            schedule_page_index[k] = v
        print(f"✅ Selected best A–D from {os.path.basename(best['set']['Schedule-A']['pdf'])}")

    if hi_candidates:
        best = min(hi_candidates, key=lambda x: x["combo_score"])
        for k, v in best["set"].items():
            schedule_page_index[k] = v
        print(f"✅ Selected best H–I from {os.path.basename(best['set']['Schedule-H']['pdf'])}")

    for k, v in schedule_page_index.items():
        if v:
            final_results[k] = {
                "page": v["page"],
                "pdf": v["pdf"]
            }

    print(final_results)
    return final_results


# =============================================================================
# Zone AB
# =============================================================================


def extract_zone_ab_details(text_chunk: str) -> str:
    prompt = f"""
              You are provided a section of a tender document related to the **Current Site Conditions**.

              📌 Summarize the overall condition in **1-2 concise lines**.

              ❌ Do NOT include layouts, section headers, markdown formatting or conversational fluff ("After reviewing...").
              🚫 Avoid long descriptions. Limit output to around 100 tokens.
              ✅ Capture the essence of the existing site in a crisp summary.
              ✅ Only mention about no. of lanes, current width, chainage, land, carrigeway. Use lines, instead of keywords. Dn not talk about bridges, ROB/RUB, grade seperator, culvert etc. at all.

              Here is the document chunk:

              {text_chunk}
              """
    return query_deepseek(prompt).strip()


def process_zone_ab(pdf_path: str, start_page: int, end_page: int, page_chunk_size: int = 5, results: dict = None) -> None:
    extracted_chunks = []

    print(f"\n🚀 Starting Zone AB Extraction\n📄 PDF: {pdf_path}\n📚 Pages: {start_page} to {end_page}")

    with pdfplumber.open(pdf_path) as pdf:
        pages = [extract_page_content(pdf.pages[i]) for i in range(start_page - 1, end_page)]

    for i in range(0, len(pages), page_chunk_size):
        chunk_text = "\n".join(pages[i:i + page_chunk_size])
        page_range = f"{start_page + i}–{min(start_page + i + page_chunk_size - 1, end_page)}"
        print(f"\n🚧 Processing pages {page_range}...")

        try:
            output = extract_zone_ab_details(chunk_text)
            print(f"✅ Summary received for pages {page_range}.")
            extracted_chunks.append(output)
        except Exception as e:
            print(f"❌ Error processing pages {page_range}: {e}")
            continue

    combined = " ".join(extracted_chunks)
    cleaned = re.sub(r"(?m)^\s*\d+\.\s*", "", combined).strip()

    results["CURRENT_SITE"] = cleaned
    print("✅ Final summarized Site Conditions and added to results.")


# =============================================================================
# 🏗️ Zone BC
# =============================================================================


class Topic(Enum):
    GEOMETRIC_DESIGN = "Geometric Designs and General Features"
    INTERSECTIONS = "Intersections, Grade Separated structures, road cuts and embankments"
    PAVEMENT = "Pavement"
    DRAINAGE = "Roadside Drainage"
    STRUCTURES = "Design of Structures like culverts, bridges, ROBs, RUBs etc.  "
    TRAFFIC_CONTROL = "Traffic Control Devices, Road Safety Works, Roadside Furniture and Compulsory afforestation"
    HAZARDS = "Hazardous locations, protection work, Special requirements for hill road (breast wall, gabion wall, toe wall, retaining wall etc.)"
    UTILITIES = "Utility, change of scope"

def parse_json_from_response(response: str) -> dict:
    """
    Extracts the first JSON object literal (starts with '{', balanced braces) from a raw LLM response.
    """

    # Try fenced block first
    match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)

    else:
        # Fallback: extract literal from first '{' to its matching '}'
        start = response.find("{")
        if start == -1:
            raise ValueError("No JSON object found in response.")
        # Find matching brace
        depth = 0
        for i, ch in enumerate(response[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = response[start:i+1]
                    break
        else:
            raise ValueError("Unbalanced braces—invalid JSON")

    return json.loads(json_str)

def init_topic_dict() -> Dict[str, List[str]]:
    return {t.name: [] for t in Topic}

def extract_zone_bc_details(text_chunk: str) -> Dict[str, str]:
    print("📤 Preparing LLM prompt...")

    prompt = f"""
              You are analyzing a tender section related to **Zone BC** that contains descriptions of civil works. These are grouped under the following topics:

              Geometric Designs and General Features
              Intersections, Grade Separated structures, road cuts and embankments
              Pavement
              Roadside Drainage
              Design of Structures like culverts, bridges, ROBs, RUBs etc.
              Traffic Control Devices, Road Safety Works, Roadside Furniture and Compulsory afforestation
              Hazardous locations, protection work, Special requirements for hill road (breast wall, gabion wall, toe wall, retaining wall etc.)
              Utility, change of scope

              ----------------------------

              🎯 **Your task**:

              - Put every detail under the **most relevant topic only**.
              - If a topic is not present in the text, **do not include it** in the JSON.
              - Do **not** nest dictionaries or use JSON objects as values — values must be plain markdown strings.
              - Give detailed breakdowns in Markdown using headings, bullet points, and tables
              **Priority Order**:
                - Tables for structured data (costs, schedules)
                - Bullet lists for descriptive fields (terrain, materials)
                - Bold labels for key-value pairs

              🚫 Strictly exclude:
              - Citing pages/sources ("As per Page 12...")
              - Conversational fluff ("After reviewing...")
              - Explain missing data (use `[Not specified]`)
              - Leaving any cells blank. Every row must explicitly include: Sub-Work, Stage ,Weightage. If a Sub-Work or Stage repeats across multiple rows, repeat it explicitly in each row.
              - using ditto marks, hyphens (-), or blank cells to imply repeated values — always fill them.

              From the input text, extract **only those items** that meet **ALL** the following rules:

              🚫 **Strictly DO NOT include** items that:

              - Are labeled or described as "NIL", "Zero", "Not specified", "not available", "not mentioned", "TBD", "indicative only", or similar.
              - Have **no dimensions, quantity, or % weightage** mentioned.
              - Have **zero weightage** or are marked as having "nil %"

              ✅ **Only include** items that:

              - Are explicitly mentioned.
              - Have valid and non-zero quantity, dimensions, or % weightage.
              - Are part of a work that has some kind of **measurable** or **quantifiable** value.

              ----------------------------

              🔎 **Important**: DO NOT include markdown fences (no ```). DO NOT MISS EVEN A SINGLE ITEM WITH NON-ZERO WORK, however small it may be
              Respond **only** in the following JSON format (no explanation, no markdown fencing, no comments):

              {{
                "Geometric Designs and General Features": "Markdown-formatted content here...",
                "Intersections, Grade Separated structures, road cuts and embankments": "Markdown-formatted content here...",
                ...
              }}

              ----------------------------

              Here is the document chunk to analyze:

              {text_chunk}
              """


    response = query_deepseek(prompt)
    print("🧠 Raw response from LLM:\n", response)

    # Extract JSON from markdown block
    try:
        parsed = parse_json_from_response(response)
    except Exception as exc:
        print("❌ JSON parse failure. Raw response was:\n", response)
        raise

    return parsed

def process_zone_bc(pdf_path: str, start_page: int, end_page: int, results: dict, page_chunk_size: int = 3) -> None:
    """
    Processes Zone BC pages and populates each topic entry directly into `results` dict.
    No return value; modifies `results` in-place.
    """
    topic_data = init_topic_dict()

    print(f"\n🚀 Starting Zone BC Extraction\n📄 PDF: {pdf_path}\n📚 Pages: {start_page} to {end_page}")
    with pdfplumber.open(pdf_path) as pdf:
        pages = [extract_page_content(pdf.pages[i]) for i in range(start_page - 1, end_page)]

    for i in range(0, len(pages), page_chunk_size):
        chunk_start = i + start_page
        chunk_end = min(i + start_page + page_chunk_size - 1, end_page)
        print(f"\n🚧 Processing page chunk {chunk_start} to {chunk_end}")

        chunk_text = "\n".join(pages[i:i + page_chunk_size])

        try:
            llm_output = extract_zone_bc_details(chunk_text)
            print(f"✅ LLM output received for pages {chunk_start}-{chunk_end}")
        except Exception as e:
            print(f"❌ Error processing pages {chunk_start}–{chunk_end}: {e}")
            continue

        for topic_str, desc in llm_output.items():
            matched_enum = next((t for t in Topic if t.value.strip() == topic_str.strip()), None)
            if matched_enum:
                print(f"📌 Appending data to topic: {matched_enum.name}")
                topic_data[matched_enum.name].append(desc.strip())
            else:
                print(f"⚠️ Unmatched topic string: '{topic_str}' — skipping")

    print("\n✅ Zone BC Processing Completed.")

    # ⬇️ Directly store in `results` dictionary
    for topic_name, entries in topic_data.items():
        results[topic_name] = entries
    print("✅ Final summarized topics and added to results.")

# =============================================================================
# 🏗️ Zone CD
# =============================================================================


def extract_zone_cd_details(text_chunk: str) -> str:
    prompt = f"""
              You are provided a section of a tender document related to Project Facilities.

              Your task is to extract only specific, actionable work details in Markdown format.

              🚫 Strictly exclude:
              - Sections marked as NIL
              - Any item without dimensions, quantities, or tabular data
              - Placeholder text like "as per manual", "details not provided", "indicative only"

              ✅ Include:
              - Only non-zero, detailed, and quantifiable works
              - Use proper Markdown formatting (headings, bullet points, tables)

              Respond ONLY with formatted Markdown. Do NOT include any intro text or explanations.

              Here is the document chunk:

              {text_chunk}
              """

    raw_response = query_deepseek(prompt)
    return raw_response.strip()

def process_zone_cd(pdf_path: str, start_page: int, end_page: int, page_chunk_size: int = 5, results: dict = None) -> None:
    extracted_chunks = []

    print(f"\n🚀 Starting Zone CD Extraction\n📄 PDF: {pdf_path}\n📚 Pages: {start_page} to {end_page}")

    with pdfplumber.open(pdf_path) as pdf:
        pages = [extract_page_content(pdf.pages[i]) for i in range(start_page - 1, end_page)]

    for i in range(0, len(pages), page_chunk_size):
        chunk_text = "\n".join(pages[i:i + page_chunk_size])
        page_range = f"{start_page + i}–{min(start_page + i + page_chunk_size - 1, end_page)}"
        print(f"\n🚧 Processing pages {page_range}...")

        try:
            output = extract_zone_cd_details(chunk_text)
            print(f"✅ LLM output received for pages {page_range}.")
            extracted_chunks.append(output)
        except Exception as e:
            print(f"❌ Error processing pages {page_range}: {e}")
            continue

    results["PROJECT_FACILITIES"] = extracted_chunks
    print("✅ Final summarized Project Facilities and added to results.")


# =============================================================================
# 🏗️ Zone HI
# =============================================================================


def extract_zone_hi_details(text_chunk: str) -> str:
    prompt = f"""
              You are provided a section of a tender document related to **Payment Terms and Weightages**.

              Your task is to extract ONLY the **tabular data showing payment weightages** for works or subworks, no need to mention payment procedure.

              ✅ Respond strictly in Markdown **table format**.
              - Do NOT include works with zero percent weightage
              - Do NOT leave any cells blank. Every row must explicitly include: Sub-Work, Stage ,Weightage. If a Sub-Work or Stage repeats across multiple rows, repeat it explicitly in each row.
              - Avoid using ditto marks, hyphens (-), or blank cells to imply repeated values — always fill them in.
              - Make a seperate table for major works, and only include their subworks in that table
              - Do not forget to include the percentge of major work in heading above table

              **Examples**:
              ### Payment Weightage
              | Work Item        | Stage          | Weightage |
              |------------------|----------------|-----------|
              | Bridge Works     | Earthwork      | 22.70%    |
              | Protection Works | Breast Wall    | 59.32%    |

              ❌ Do NOT include any headings, intro text, notes, explanations, or layout details, citations or conversational fluff ("After reviewing...").

              Here is the document chunk:

              {text_chunk}
              """
    return query_deepseek(prompt).strip()


def process_zone_hi(pdf_path: str, start_page: int, end_page: int, page_chunk_size: int = 10, results: dict = None) -> None:
    extracted_chunks = []

    print(f"\n🚀 Starting Zone HI Extraction\n📄 PDF: {pdf_path}\n📚 Pages: {start_page} to {end_page}")

    with pdfplumber.open(pdf_path) as pdf:
        pages = [extract_page_content(pdf.pages[i]) for i in range(start_page - 1, end_page)]

    for i in range(0, len(pages), page_chunk_size):
        chunk_text = "\n".join(pages[i:i + page_chunk_size])
        page_range = f"{start_page + i}–{min(start_page + i + page_chunk_size - 1, end_page)}"
        print(f"\n🚧 Processing pages {page_range}...")

        try:
            output = extract_zone_hi_details(chunk_text)
            print(f"✅ Payment table extracted for pages {page_range}.")
            extracted_chunks.append(output)
        except Exception as e:
            print(f"❌ Error processing pages {page_range}: {e}")
            continue

    results["PAYMENT_WEIGHTAGE"] = extracted_chunks
    print("✅ Final Payment Weightage tables added to results.")


# =============================================================================
# Image Analysis
# =============================================================================


# Function to render a PDF page to an image
def render_page_to_image(page) -> Image.Image:
    return page.get_pixmap(dpi=150).pil_tobytes(format="PNG")

# Function to analyze a single page with Groq
def analyze_full_page_with_groq(pil_image_bytes: bytes) -> str:
    img_base64 = base64.b64encode(pil_image_bytes).decode("utf-8")
    image_data_url = f"data:image/png;base64,{img_base64}"

    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is a page from an Indian tender Schedule B. Extract all useful content from it. Format it into readable markdown tables and bullet points. Include data from drawings, tables, or other visual information."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    }
                ]
            }
        ],
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content.strip()

# Function to check if a PDF page has images
def page_has_images(page) -> bool:
    return len(page.get_images(full=True)) > 0

# Main driver function
def extract_zone_bc_image_info(pdf_path: str, start_page: int, end_page: int, results: dict = None) -> None:
    doc = fitz.open(pdf_path)
    all_responses = []

    print(f"\n🖼️ Extracting image-based info from Zone BC pages {start_page} to {end_page}")

    for page_num in range(start_page - 1, end_page):
        page = doc.load_page(page_num)
        if page_has_images(page):
            print(f"📸 Analyzing Page {page_num + 1} (has images)...")
            image_bytes = render_page_to_image(page)
            try:
                response = analyze_full_page_with_groq(image_bytes)
                all_responses.append(f"\n### Page {page_num + 1}\n" + response)
            except Exception as e:
                print(f"❌ Error analyzing Page {page_num + 1}: {e}")

    combined_text = "\n".join(all_responses)

    if combined_text.strip():
        print("\n🧠 Sending extracted image data for final summarization...")

        final_completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Based on the extracted information below, give detailed info on everything specifically related "
                        "to **TCS** , **Road Works** and **Pavement** in structured markdown format. Only mention the **non-zero works**:\n\n"
                        + combined_text
                    )
                }
            ],
            temperature=0.4,
            max_completion_tokens=2048,
            top_p=1,
            stream=False,
        )

        results["IMAGE_SUMMARY"] = final_completion.choices[0].message.content.strip()
        print("✅ Final summarized TCS and Road Works content added to results.")
    else:
        print("⚠️ No valid image content found. Nothing to summarize.")


# =============================================================================
# Processing
# =============================================================================


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


