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

logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger("fitz").setLevel(logging.ERROR)

# Groq Setup
groq_client = Groq(api_key="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")

from utils import query_deepseek
from utils import extract_page_content


# =============================================================================
# 📦 Retriever Functions
# The following functions are used to extract, filter, and match document chunks
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
        "Schedule-D": {}
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

    for root, dirs, files in os.walk(folder_path):
        logging.info(f"📂 Scanning directory: {root}")
        for filename in files:
            if not filename.lower().endswith(".pdf") or filename.startswith("._"):
                logging.info(f"⏭️ Skipping file: {filename}")
                continue

            full_path = os.path.join(root, filename)
            logging.info(f"📄 Processing file: {filename}")

            try:
                all_chunks = extract_chunks_with_metadata(full_path)
            except Exception as e:
                logging.error(f"⚠️ Error reading {filename}: {e}")
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
                    logging.warning(f"❌ {sched} not found in {filename}")
                    all_found = False
                    break

            if all_found:
                pages = [abcd_found[s]["page"] for s in ["Schedule-A", "Schedule-B", "Schedule-C", "Schedule-D"]]
                if sorted(pages) == pages and len(set(v["pdf"] for v in abcd_found.values())) == 1:
                    abcd_candidates.append({"set": abcd_found, "combo_score": combo_score})
                    logging.info(f"✅ Found A–D in order in {filename} | Combo Score: {combo_score}")
                else:
                    logging.warning(f"⚠️ A–D found, but not in order | Combo Score: {combo_score}")

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
                    logging.info(f"✅ Found Notice Inviting Bid in {filename} on page {page}")

    if not abcd_candidates:
        logging.error("❌ Schedule A–D could not be found in any PDF. Aborting.")
        raise ValueError("Schedule A–D not found")

    # Choose best A-D candidate
    best = min(abcd_candidates, key=lambda x: x["combo_score"])
    for k, v in best["set"].items():
        schedule_page_index[k] = v
    logging.info(f"✅ Selected best A–D from {os.path.basename(best['set']['Schedule-A']['pdf'])}")

    for k, v in schedule_page_index.items():
        if v:
            final_results[k] = {
                "page": v["page"],
                "pdf": v["pdf"]
            }

    logging.info(f"📋 Final Results: {json.dumps(final_results, indent=2)}")
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
    logger.info(f"🚀 [Zone AB] Starting Extraction | File: {pdf_path} | Pages: {start_page}-{end_page}")

    with pdfplumber.open(pdf_path) as pdf:
        pages = [extract_page_content(pdf.pages[i]) for i in range(start_page - 1, end_page)]

    for i in range(0, len(pages), page_chunk_size):
        chunk_text = "\n".join(pages[i:i + page_chunk_size])
        page_range = f"{start_page + i}–{min(start_page + i + page_chunk_size - 1, end_page)}"
        logger.info(f"🚧 [Zone AB] Processing pages {page_range}...")

        try:
            output = extract_zone_ab_details(chunk_text)
            logger.info(f"✅ [Zone AB] Summary extracted for pages {page_range}")
            extracted_chunks.append(output)
        except Exception as e:
            logger.error(f"❌ [Zone AB] Error processing pages {page_range}: {e}")
            continue

    combined = " ".join(extracted_chunks)
    cleaned = re.sub(r"(?m)^\s*\d+\.\s*", "", combined).strip()
    results["CURRENT_SITE"] = cleaned
    logger.info("✅ [Zone AB] Site Conditions summary stored.")


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
    logger.debug("🔍 Attempting to parse JSON from LLM response...")
    match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        logger.debug("✅ Found fenced JSON block.")
        json_str = match.group(1)
    else:
        logger.debug("⚠️ No fenced block. Falling back to brace matching...")
        start = response.find("{")
        if start == -1:
            logger.error("❌ No JSON object found in response.")
            raise ValueError("No JSON object found in response.")
        depth = 0
        for i, ch in enumerate(response[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = response[start:i + 1]
                    logger.debug("✅ Found JSON using brace matching.")
                    break
        else:
            logger.error("❌ Unbalanced braces—invalid JSON")
            raise ValueError("Unbalanced braces—invalid JSON")

    try:
        parsed_json = json.loads(json_str)
        logger.debug("✅ JSON parsing successful.")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decode error: {e}")
        raise

def init_topic_dict() -> Dict[str, List[str]]:
    logger.debug("📦 Initializing topic dictionary...")
    return {t.name: [] for t in Topic}

def extract_zone_bc_details(text_chunk: str) -> Dict[str, str]:
    logger.info("📤 Preparing LLM prompt for Zone BC extraction...")

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

    logger.debug("📨 Sending prompt to LLM...")
    response = query_deepseek(prompt)
    logger.info("🧠 Received raw response from LLM.")
    logger.debug(f"🧾 Response Preview:\n{response[:500]}...")

    try:
        parsed = parse_json_from_response(response)
        logger.info("✅ Parsed JSON successfully from LLM response.")
    except Exception as exc:
        logger.error(f"❌ JSON parse failure: {exc}")
        raise

    return parsed

def process_zone_bc(pdf_path: str, start_page: int, end_page: int, results: dict, page_chunk_size: int = 3) -> None:
    topic_data = init_topic_dict()
    logger.info(f"🚀 [Zone BC] Starting extraction | File: {pdf_path} | Pages: {start_page}-{end_page}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [extract_page_content(pdf.pages[i]) for i in range(start_page - 1, end_page)]
        logger.debug("📄 PDF pages successfully extracted.")
    except Exception as e:
        logger.error(f"❌ Failed to open or read PDF: {e}")
        return

    for i in range(0, len(pages), page_chunk_size):
        chunk_start = i + start_page
        chunk_end = min(i + start_page + page_chunk_size - 1, end_page)
        logger.info(f"🚧 [Zone BC] Processing page chunk {chunk_start}–{chunk_end}")

        chunk_text = "\n".join(pages[i:i + page_chunk_size])
        try:
            llm_output = extract_zone_bc_details(chunk_text)
            logger.info(f"✅ [Zone BC] Output received for pages {chunk_start}-{chunk_end}")
        except Exception as e:
            logger.error(f"❌ [Zone BC] Error processing pages {chunk_start}–{chunk_end}: {e}")
            continue

        for topic_str, desc in llm_output.items():
            matched_enum = next((t for t in Topic if t.value.strip() == topic_str.strip()), None)
            if matched_enum:
                logger.info(f"📌 [Zone BC] Appending data to topic: {matched_enum.name}")
                topic_data[matched_enum.name].append(desc.strip())
            else:
                logger.warning(f"⚠️ [Zone BC] Unmatched topic string: '{topic_str}' — skipping")

    for topic_name, entries in topic_data.items():
        results[topic_name] = entries
        logger.debug(f"📝 [Zone BC] Added {len(entries)} entries under topic: {topic_name}")

    logger.info("✅ [Zone BC] All topics summarized and added to results.")


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
    logger.info(f"🚀 [Zone CD] Starting Extraction | File: {pdf_path} | Pages: {start_page}-{end_page}")

    with pdfplumber.open(pdf_path) as pdf:
        pages = [extract_page_content(pdf.pages[i]) for i in range(start_page - 1, end_page)]

    for i in range(0, len(pages), page_chunk_size):
        chunk_text = "\n".join(pages[i:i + page_chunk_size])
        page_range = f"{start_page + i}–{min(start_page + i + page_chunk_size - 1, end_page)}"
        logger.info(f"🚧 [Zone CD] Processing pages {page_range}...")

        try:
            output = extract_zone_cd_details(chunk_text)
            logger.info(f"✅ [Zone CD] LLM output received for pages {page_range}")
            extracted_chunks.append(output)
        except Exception as e:
            logger.error(f"❌ [Zone CD] Error processing pages {page_range}: {e}")
            continue

    results["PROJECT_FACILITIES"] = extracted_chunks
    logger.info("✅ [Zone CD] Final Project Facilities added to results.")


# =============================================================================
# 📸 Image Analysis
# =============================================================================


def render_page_to_image(page) -> bytes:
    return page.get_pixmap(dpi=150).pil_tobytes(format="PNG")


def analyze_full_page_with_groq(pil_image_bytes: bytes) -> str:
    img_base64 = base64.b64encode(pil_image_bytes).decode("utf-8")
    image_data_url = f"data:image/png;base64,{img_base64}"

    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a page from an Indian tender Schedule B. Extract all useful content from it. Format it into readable markdown tables and bullet points. Include data from drawings, tables, or other visual information."},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content.strip()


def page_has_images(page) -> bool:
    return len(page.get_images(full=True)) > 0


def extract_zone_bc_image_info(pdf_path: str, start_page: int, end_page: int, results: dict = None) -> None:
    doc = fitz.open(pdf_path)
    all_responses = []

    logging.info(f"🖼️ Extracting image-based info from Zone BC pages {start_page}–{end_page}")

    for page_num in range(start_page - 1, end_page):
        page = doc.load_page(page_num)
        if page_has_images(page):
            logging.info(f"📸 Analyzing Page {page_num + 1} (has images)...")
            try:
                image_bytes = render_page_to_image(page)
                response = analyze_full_page_with_groq(image_bytes)
                all_responses.append(f"\n### Page {page_num + 1}\n{response}")
            except Exception as e:
                logging.error(f"❌ Error analyzing Page {page_num + 1}: {e}")

    combined_text = "\n".join(all_responses)

    if combined_text.strip():
        logging.info("🧠 Sending extracted image data for final summarization...")

        try:
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
            logging.info("✅ Final summarized TCS and Road Works content added to results.")
        except Exception as e:
            logging.error(f"❌ Error during final summarization: {e}")
    else:
        logging.warning("⚠️ No valid image content found. Nothing to summarize.")
