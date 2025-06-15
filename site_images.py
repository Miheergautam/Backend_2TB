import requests
import base64
from PIL import Image
from io import BytesIO
from groq import Groq

# CONFIGS
SERPAPI_API_KEY = "ee9869f199c55efdc0ae10df13c2d11b2028c7baf194ef856ab88bd00cf6822a"
DEEPSEEK_API_KEY = "sk-fe754eb8e5a04ec79de5c71064b5e25d"
groq_client = Groq(api_key="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")

def extract_locations_from_description(description):
    print("📌 Extracting locations from description...")

    prompt = f"""Extract only the two most relevant location points (start and end) from the following road description.
Return them as a JSON list of two strings like ["Start", "End"] which will be locations. If there is just one location, keep that as both start and end. Remember both start and end should be just location like 'Kargil', just one word.

Road description:
{description}
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", json=payload, headers=headers)
        content = response.json()["choices"][0]["message"]["content"]
        print("🧾 DeepSeek raw response:", content)

        # Extract JSON list like ["Kargil", "Dumgil"]
        match = re.search(r"\[.*?\]", content)
        if not match:
            raise ValueError("No valid location list found in response.")

        locations = json.loads(match.group(0))

        if len(locations) == 1:
            locations = [locations[0], locations[0]]

        print("✅ Locations extracted:", locations)
        return locations if len(locations) == 2 else []

    except Exception as e:
        print(f"❌ DeepSeek Error: {e}")
        return []


# Fetch images via SerpAPI
def get_road_location_images(query, count=3):
    print(f"🔍 Fetching {count} images for: {query}")
    params = {
        "engine": "google_images",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "google_domain": "google.co.in",
        "gl": "in",
        "hl": "en",
        "ijn": "0",
        "tbm": "isch"
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        image_urls = [
            img.get("original") or img.get("thumbnail") or img.get("source")
            for img in data.get("images_results", [])[:count]
        ]
        print(f"✅ Retrieved {len(image_urls)} images.")
        return image_urls
    except Exception as e:
        print(f"❌ Failed to fetch images for {query}: {e}")
        return []

# Filter valid images, separate Instagram
def filter_and_fetch_images(image_urls):
    print("🧹 Filtering and downloading image content...")
    photos, insta = [], []
    for url in image_urls:
        if "instagram.com" in url:
            insta.append(url)
            continue
        try:
            res = requests.get(url, timeout=5)
            res.raise_for_status()
            img = Image.open(BytesIO(res.content))
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            photos.append({"url": url, "base64": base64_image})
        except Exception:
            print(f"⚠️ Skipped invalid image: {url}")
            continue
    print(f"✅ {len(photos)} valid photos, {len(insta)} Instagram links.")
    return photos[:5], insta[:2]  # Limit Instagram to 2

# Use Groq LLaMA-4 Vision to rank and select best images
def select_best_images_groq(photos, insta, start_loc, end_loc):
    print(f"🧠 Passing {len(photos)} photos to Groq for selection...")

    images_payload = []
    text_prompt = f"""
These are aerial photos of a road's start and end points:

- Start: {start_loc}
- End: {end_loc}

Select and rank the 4 most relevant photos (1 = most relevant). Base your judgment on terrain, road layout, and contextual relevance.
Return a ranked list of indexes like [3, 1, 4, 2].
Only consider the following {len(photos)} non-Instagram images.
"""

    for i, img in enumerate(photos):
        images_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}"}
        })

    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text_prompt}] + images_payload
                }
            ],
            temperature=0.4,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
        )
        text_response = response.choices[0].message.content.strip()
        print("📝 Groq raw response:", text_response)

        # Parse and validate indexes
        indexes = re.findall(r"\d+", text_response.splitlines()[-1])
        indexes = list(map(int, indexes))

        # Select photo URLs based on Groq's response
        selected_photos = [photos[i-1]["url"] for i in indexes if 1 <= i <= len(photos)]

        # Concat Instagram first, then Groq-ranked photos, trimmed to max 4
        combined = insta + selected_photos
        final_output = combined[:4]

        print(f"🎯 Groq selected {len(selected_photos)} photos. Total final images: {len(final_output)}")
        return final_output

    except Exception as e:
        print("❌ Groq error:", e)
        fallback = insta + [p["url"] for p in photos[:(4 - len(insta))]]
        print("⚠️ Using fallback:", fallback[:4])
        return fallback[:4]


# Main pipeline
def process_and_display_images(road_description, state="Madhya Pradesh"):
    print("🚦 Starting image processing pipeline...")
    locations = extract_locations_from_description(road_description)
    if len(locations) < 2:
        print("❌ Failed to extract valid locations.")
        return

    queries = [f"{loc}, {state} aerial drone view" for loc in locations]
    all_urls = []
    for query in queries:
        all_urls.extend(get_road_location_images(query))

    photos, insta = filter_and_fetch_images(all_urls)
    if not photos and not insta:
        print("❌ No usable images found.")
        return

    final_images = select_best_images_groq(photos, insta, locations[0], locations[1])
    # print("\n📸 Final 4 Selected Image URLs:")
    
    return final_images

