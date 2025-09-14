# app.py
# --------------------------------------------
# Upload 4 images in one go -> identify each via OpenAI gpt-4o (vision)
# -> aggregate top-3 overall -> user picks one (or "Other")
# -> fetch price + description + sources (with web_search)
# -> classify item name into a Craigslist category.
#
# Setup:
#   pip install flask "openai>=1.40.0" python-multipart
#   export OPENAI_API_KEY=your_key_here
#   python app.py
#
# Debugging:
# - The page shows a "Debug (expand)" panel with:
#     * Per-image request metadata (no raw image data)
#     * The JSON schema we ask the model to return
#     * The raw JSON returned (or the exception)
#     * Final "price+description" request + raw response
#     * Category classification request + raw response
# --------------------------------------------

import os, time, json, base64, uuid, re
from collections import defaultdict
from flask import Flask, request, render_template_string
from openai import OpenAI
from datetime import datetime

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Create listings directory
LISTINGS_DIR = "listings"
if not os.path.exists(LISTINGS_DIR):
    os.makedirs(LISTINGS_DIR)

# Craigslist category list (single source of truth)
CATEGORIES = [
    "antiques",
    "appliances",
    "arts & crafts",
    "atvs, utvs, snowmobiles",
    "auto parts",
    "auto wheels & tires",
    "aviation",
    "baby & kid stuff",
    "barter",
    "bicycle parts",
    "bicycles",
    "boat parts",
    "boats",
    "books & magazines",
    "business/commercial",
    "cars & trucks ($5)",
    "cds / dvds / vhs",
    "cell phones",
    "clothing & accessories",
    "collectibles",
    "computer parts",
    "computers",
    "electronics",
    "farm & garden",
    "free stuff",
    "furniture",
    "garage & moving sales",
    "general for sale",
    "health and beauty",
    "heavy equipment",
    "household items",
    "jewelry",
    "materials",
    "motorcycle parts",
    "motorcycles/scooters ($5)",
    "musical instruments",
    "photo/video",
    "rvs ($5)",
    "sporting goods",
    "tickets",
    "tools",
    "toys & games",
    "trailers",
    "video gaming",
    "wanted"
]

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Identify an Item from 4 Photos</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:980px;margin:40px auto;padding:0 16px}
    .card{border:1px solid #e5e7eb;border-radius:16px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
    .grid{display:grid;gap:12px}
    .grid-2{grid-template-columns:repeat(2,minmax(0,1fr))}
    .muted{color:#6b7280}
    button{padding:10px 16px;border-radius:10px;border:1px solid #111827;background:#111827;color:#fff;cursor:pointer}
    input[type=file]{border:1px solid #d1d5db;border-radius:10px;padding:10px;width:100%}
    select, input[type=text], textarea{border:1px solid #d1d5db;border-radius:8px;padding:10px;width:100%}
    details summary{cursor:pointer}
    .hint{font-size:12px;color:#6b7280}
    a{color:#2563eb;text-decoration:none}
    pre{background:#0b1021;color:#e5e7eb;padding:12px;border-radius:12px;overflow:auto}
    code{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;}
    .warn{color:#b45309}
    label b {display:inline-block;margin-bottom:6px}
  </style>
</head>
<body>
  <h1>Identify an Item from 4 Photos</h1>
  <p class="muted">Upload 4 images. We’ll analyze each photo with OpenAI’s best model, merge the top guesses, and then fetch a price + short description.</p>

  <form class="card" method="POST" action="/analyze" enctype="multipart/form-data">
    <label><b>Images (choose 4)</b></label>
    <input type="file" name="images" accept="image/*" multiple required>
    <p class="hint">Select exactly four images in one go. We only process these in-memory.</p>
    <button type="submit">Analyze</button>
  </form>

  {% if error_msg %}
    <div class="card">
      <p class="warn"><b>Error:</b> {{ error_msg }}</p>
    </div>
  {% endif %}

  {% if combined %}
    <div class="card">
      <h2>Combined Top 3 Guesses</h2>
      <form method="POST" action="/choose">
        {% for row in combined %}
          <div>
            <input type="radio" id="g{{ loop.index0 }}" name="choice" value="{{ row['label'] }}" {% if loop.first %}checked{% endif %}>
            <label for="g{{ loop.index0 }}"><b>{{ row['label'] }}</b> — weight {{ '%.3f' % row['weight'] }}</label>
          </div>
        {% endfor %}
        <div style="margin-top:10px">
          <input type="radio" id="other" name="choice" value="__other__">
          <label for="other">Other:</label>
          <input type="text" name="other_text" placeholder="Brand + model">
        </div>
        <input type="hidden" name="session_payload" value='{{ session_payload|tojson }}'>
        <button type="submit" style="margin-top:12px">Get Price & Description</button>
      </form>
    </div>

    <div class="card">
      <details>
        <summary>Per-image top-3 (expand)</summary>
        {% for guesses in per_image %}
          <p><b>Image {{ loop.index }}</b></p>
          <ul>
          {% for g in guesses %}
            <li>{{ g['label'] }} — conf {{ '%.3f' % g['confidence'] }}</li>
          {% endfor %}
          </ul>
        {% endfor %}
      </details>
    </div>

    {% if debug_entries %}
      <div class="card">
        <details open>
          <summary><b>Debug (expand)</b></summary>
          {% for d in debug_entries %}
            <p><b>{{ d.title }}</b></p>
            <pre><code>{{ d.payload | tojson(indent=2) }}</code></pre>
            {% if not loop.last %}<hr>{% endif %}
          {% endfor %}
        </details>
      </div>
    {% endif %}
  {% endif %}

  {% if result %}
    <div class="card">
      <h2>{{ result['label'] }}</h2>
      
      <form method="POST" action="/save_listing">
        <input type="hidden" name="listing_id" value="{{ result.get('listing_id', '') }}">
        <input type="hidden" name="label" value="{{ result['label'] }}">
        <input type="hidden" name="market_price" value="{{ result['market_price'] }}">
        <input type="hidden" name="session_payload" value='{{ session_payload|tojson }}'>
        
        <p><b>Craigslist Description:</b></p>
        <textarea name="description" style="width:100%;min-height:120px">{{ result['description'] }}</textarea>

        <div class="grid grid-2" style="margin:16px 0">
          <div>
            <label for="category"><b>Category</b></label>
            <select id="category" name="category">
              {% for cat in categories %}
                <option value="{{ cat }}" {% if cat == result.get('category','') %}selected{% endif %}>{{ cat }}</option>
              {% endfor %}
            </select>
            <p class="hint">Pre-selected by ChatGPT. Please change if needed.</p>
          </div>
          <div>
            <label><b>Market Price:</b></label>
            <div style="font-weight:bold">{{ result['market_price'] }}</div>
            <label for="selling_price" style="margin-top:12px"><b>Selling Price:</b></label>
            <input type="text" id="selling_price" name="selling_price" value="{{ result['selling_price'] }}" style="font-weight:bold;color:#059669">
          </div>
        </div>
        
        <button type="submit" style="background:#059669;margin-right:12px">Save Listing</button>
        <a href="/" style="color:#6b7280;text-decoration:none">Start over</a>
      </form>
      
      {% if result['sources'] %}
        <details style="margin-top:16px">
          <summary class="muted">Sources</summary>
          <ul>
            {% for s in result['sources'] %}
              <li><a href="{{ s['url'] }}" target="_blank" rel="noopener">{{ s['title'] }}</a></li>
            {% endfor %}
          </ul>
        </details>
      {% endif %}
    </div>

    {% if debug_entries %}
      <div class="card">
        <details open>
          <summary><b>Debug (expand)</b></summary>
          {% for d in debug_entries %}
            <p><b>{{ d.title }}</b></p>
            <pre><code>{{ d.payload | tojson(indent=2) }}</code></pre>
            {% if not loop.last %}<hr>{% endif %}
          {% endfor %}
        </details>
      </div>
    {% endif %}
  {% endif %}
  
  {% if saved_listing %}
    <div class="card" style="background:#f0fdf4;border-color:#22c55e">
      <h2>✅ Listing Saved!</h2>
      <p><b>Listing ID:</b> {{ saved_listing['id'] }}</p>
      <p><b>Files saved:</b> {{ saved_listing['files']|join(', ') }}</p>
      <p><b>Location:</b> {{ saved_listing['path'] }}</p>
      <a href="/">Create another listing</a>
    </div>
  {% endif %}
</body>
</html>
"""

def canonicalize(label: str) -> str:
    """Basic canonical label for dedupe (lowercase, strip, squash spaces)."""
    return " ".join((label or "").lower().strip().split())

class DebugEntry:
    def __init__(self, title: str, payload):
        self.title = title
        self.payload = payload

def ask_o3_for_top3(image_bytes: bytes, filename: str, mime: str, debug_list):
    """
    Send ONE image to vision model and ask for top-3 JSON.
    First tries gpt-4o (vision), then falls back to gpt-4o-mini.
    Append structured debug info to debug_list.
    """
    schema = {
        "type": "object",
        "properties": {
            "guesses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["label", "confidence"],
                    "additionalProperties": False
                },
                "minItems": 3,
                "maxItems": 3
            }
        },
        "required": ["guesses"],
        "additionalProperties": False
    }

    system = (
        "You identify retail products from a single image.\n"
        "Return EXACT JSON with keys: guesses: [{label, confidence} x3].\n"
        "Label must be 'Brand Model Variant' if possible (e.g., 'Meta Quest Pro').\n"
        "Confidence is 0..1. If unsure, include best-guess labels with lower confidence."
    )
    prompt = "Identify the item in this photo. Return your top 3 distinct guesses with confidences."

    # Encode image as base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"data:{mime or 'image/jpeg'};base64,{image_base64}"

    debug_list.append(DebugEntry(
        "Vision request (gpt-4o)",
        {
            "model": "gpt-4o",
            "schema": schema,
            "system_prompt_excerpt": system[:240],
            "user_prompt": prompt,
            "image_meta": {"filename": filename or "upload", "mimetype": mime, "bytes": len(image_bytes)},
        }
    ))

    # Try GPT-4o with vision
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw = resp.choices[0].message.content
            debug_list.append(DebugEntry("Vision raw JSON (gpt-4o)", {"attempt": attempt + 1, "raw": try_parse_json(raw)}))
            data = json.loads(raw)
            return data.get("guesses", [])
        except Exception as e:
            debug_list.append(DebugEntry("Vision error (gpt-4o)", {"attempt": attempt + 1, "error": repr(e)}))
            if attempt == 0:
                time.sleep(0.8)
                continue

    # Fallback to gpt-4o-mini (text-only heuristic)
    debug_list.append(DebugEntry(
        "Vision request (fallback gpt-4o-mini)",
        {"model": "gpt-4o-mini", "note": "text-only fallback"}
    ))
    try:
        fallback_prompt = (
            "I cannot process the image, but I need to provide 3 product guesses with confidence scores. "
            "Based on common retail products, provide 3 generic guesses in JSON format: "
            "{\"guesses\": [{\"label\": \"Generic Product\", \"confidence\": 0.1}, ...]}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": fallback_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        raw = resp.choices[0].message.content
        debug_list.append(DebugEntry("Vision raw JSON (gpt-4o-mini)", {"raw": try_parse_json(raw)}))
        data = json.loads(raw)
        return data.get("guesses", [])
    except Exception as e2:
        debug_list.append(DebugEntry("Vision error (gpt-4o-mini)", {"error": repr(e2)}))
        return []

def try_parse_json(raw_text: str):
    """Best-effort to pretty capture JSON text."""
    try:
        return json.loads(raw_text)
    except Exception:
        return {"non_json_text": raw_text}

def extract_json_fallback(text: str):
    """
    Robustly extract the first JSON object from an arbitrary string.
    - Tries fenced ```json blocks first.
    - Then scans for the first balanced {...} and parses it.
    Returns dict or {}.
    """
    if not text:
        return {}

    # 1) Try to find a fenced ```json ... ``` block (or plain ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            # sometimes the block contains extra prose; try to find first balanced object inside the block
            inner = candidate

            # fallthrough to balanced-scan on the block
            text = inner

    # 2) Balanced-brace scan on the whole text
    starts = [m.start() for m in re.finditer(r"\{", text)]  # <-- fixed: look for literal "{"
    for si in starts:
        depth = 0
        for ei in range(si, len(text)):
            ch = text[ei]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[si:ei+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    return {}

def classify_category_with_llm(label: str, debug_entries):
    """
    Ask ChatGPT (Responses API) to choose ONE category from CATEGORIES for the given label.
    Returns a string category (always one of CATEGORIES), with a few keyword fallbacks.
    """
    system = (
        "You are a marketplace category classifier for Craigslist.\n"
        "Given only an item name, choose the SINGLE closest category from the provided list.\n"
        "Strict rules:\n"
        "- Output strict JSON: {\"category\": \"<one from list>\"}\n"
        "- Only choose from the provided list EXACTLY as written.\n"
        "- Prefer the most specific category available.\n"
        "- If none clearly fits, choose 'general for sale'.\n"
        "- Use common sense about prohibited items, avoiding scams, and recall information, but still return exactly one of the provided categories.\n"
    )
    cat_list = "\n".join(f"- {c}" for c in CATEGORIES)
    user = f"Item name: {label}\n\nChoose a category from this list:\n{cat_list}\n\nReturn JSON only."

    debug_entries.append(DebugEntry("Category classification request", {
        "model": "gpt-4.1-mini",
        "system_prompt_excerpt": system[:260],
        "user_prompt": user
    }))

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0
        )
        raw_text = getattr(resp, "output_text", None)
        debug_entries.append(DebugEntry("Category classification raw", {"raw_text": raw_text}))

        data = extract_json_fallback(raw_text or "")
        cat = (data.get("category") or "").strip().lower()
        for c in CATEGORIES:
            if c.lower() == cat:
                return c
    except Exception as e:
        debug_entries.append(DebugEntry("Category classification error", {"error": repr(e)}))

    # Heuristic fallback
    l = (label or "").lower()
    def has(*words): return any(w in l for w in words)

    if has("chair","sofa","table","dresser","stool","couch","desk","bed","cabinet"): return "furniture"
    if has("wegner","wishbone","eames","vitra"): return "furniture"
    if has("iphone","samsung","pixel","android","smartphone","cell"): return "cell phones"
    if has("gpu","motherboard","ssd","ram","graphics card"): return "computer parts"
    if has("laptop","macbook","surface","notebook"): return "computers"
    if has("camera","lens","dslr","mirrorless","tripod"): return "photo/video"
    if has("guitar","piano","keyboard","drum","synth"): return "musical instruments"
    if has("ps5","xbox","nintendo","switch","gaming"): return "video gaming"
    if has("bicycle","bike"): return "bicycles"
    if has("stroller","crib","kid","baby"): return "baby & kid stuff"
    if has("sofa","rug","lamp","decor"): return "household items"
    if has("jacket","jeans","shirt","dress","shoes"): return "clothing & accessories"
    if has("necklace","ring","bracelet","earring"): return "jewelry"
    if has("mower","tractor","seed","soil","planter"): return "farm & garden"
    if has("microwave","fridge","refrigerator","washer","dryer","oven","dishwasher"): return "appliances"
    if has("hammer","drill","saw","wrench","tool"): return "tools"
    if has("antique","vintage","mid-century"): return "antiques"
    if has("collectible","limited edition","trading card","funko","lego set"): return "collectibles"
    if has("speaker","headphones","tv","monitor","tablet","smartwatch"): return "electronics"

    return "general for sale"

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, categories=CATEGORIES)

@app.route("/analyze", methods=["POST"])
def analyze():
    debug_entries = []
    images = request.files.getlist("images")
    if not images or len(images) != 4:
        return render_template_string(
            INDEX_HTML,
            error_msg="Please select exactly 4 images in a single selection.",
            combined=None, per_image=None, session_payload=None, debug_entries=[],
            categories=CATEGORIES
        )

    # Store images for later saving
    session_images = []
    per_image = []
    for f in images:
        img_bytes = f.read()  # in-memory only
        session_images.append({
            'filename': f.filename,
            'mimetype': f.mimetype,
            'bytes': base64.b64encode(img_bytes).decode('utf-8')
        })
        guesses = ask_o3_for_top3(img_bytes, f.filename, f.mimetype, debug_entries) or []
        clean = []
        for g in guesses[:3]:
            try:
                label = str(g["label"])[:140]
                conf = float(g["confidence"])
                conf = max(0.0, min(conf, 1.0))
                clean.append({"label": label, "confidence": conf})
            except Exception as e:
                debug_entries.append(DebugEntry("Clean guess error", {"raw_guess": g, "error": repr(e)}))
                continue
        while len(clean) < 3:
            clean.append({"label": "Unknown", "confidence": 0.0})
        per_image.append(clean)

    # Combine: sum normalized weights across images per canonical label
    weights = defaultdict(float)
    label_map = {}
    for guesses in per_image:
        s = sum(g["confidence"] for g in guesses) or 1.0
        for g in guesses:
            w = g["confidence"] / s
            key = canonicalize(g["label"])
            weights[key] += w
            label_map.setdefault(key, g["label"])

    combined = [{"label": label_map[k], "weight": v} for k, v in weights.items()]
    combined.sort(key=lambda x: x["weight"], reverse=True)
    combined = combined[:3]

    session_payload = {"per_image": per_image, "combined": combined, "images": session_images}

    # Log aggregation for debug
    debug_entries.append(DebugEntry("Aggregation", {
        "per_image": per_image,
        "weights": {k: v for k, v in weights.items()},
        "combined": combined
    }))

    return render_template_string(
        INDEX_HTML,
        combined=combined,
        per_image=per_image,
        session_payload=session_payload,
        debug_entries=debug_entries,
        categories=CATEGORIES
    )

@app.route("/choose", methods=["POST"])
def choose():
    debug_entries = []
    choice = request.form.get("choice")
    other = (request.form.get("other_text") or "").strip()
    label = other if (choice == "__other__" and other) else choice or "Unknown"
    
    # Generate unique listing ID for this session
    listing_id = str(uuid.uuid4())[:8]

    # --- 1) Pricing + description via Responses API with web_search tool ---
    price_system = (
        "You are a Craigslist listing generator.\n"
        "TASKS:\n"
        "1) Use the web_search tool to find the current US market price for the given product name from multiple credible sources "
        "(e.g., retailer sites, marketplaces, auction results). Return the LOWEST reasonable current price you find.\n"
        "2) Write a short Craigslist-style description (2-4 sentences) that sounds like a real person, covering key features and assumed condition (good/excellent).\n\n"
        "OUTPUT: Return STRICT JSON with keys:\n"
        "{\n"
        '  "label": string,\n'
        '  "market_price": number,\n'
        '  "description": string,\n'
        '  "sources": [ {"title": string, "url": string} ]\n'
        "}\n"
        "NOTES:\n"
        "- Always invoke web_search to gather prices and include at least 2 sources with titles+URLs.\n"
        "- market_price must be a number (no currency symbol). If prices vary, choose the lowest fair price for a used-but-good item.\n"
        "- Respond with JSON only. No prose outside JSON.\n"
    )
    price_user = f"Product: {label}. Return JSON exactly as specified."

    debug_entries.append(DebugEntry("Price+Description request", {
        "model": "gpt-4.1",
        "tools": ["web_search"],
        "system_prompt_excerpt": price_system[:300],
        "user_prompt": price_user
    }))

    result = {}
    try:
        resp = client.responses.create(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": price_system},
                {"role": "user", "content": price_user}
            ],
            tools=[{"type": "web_search"}],
            temperature=0
        )
        raw_text = getattr(resp, "output_text", None)
        debug_entries.append(DebugEntry("Price+Description raw", {"raw_text": raw_text}))

        data = extract_json_fallback(raw_text or "")

        lbl = str(data.get("label") or label)
        mk = data.get("market_price")
        try:
            market_price_num = float(mk)
        except Exception:
            market_price_num = None

        descr = str(data.get("description") or "").strip()
        sources = data.get("sources") or []
        clean_sources = []
        for s in sources[:5]:
            try:
                title = str(s.get("title") or "")[:140]
                url = str(s.get("url") or "")
                if title and url:
                    clean_sources.append({"title": title, "url": url})
            except Exception:
                continue

        selling_price = "N/A"
        market_price_label = "N/A"
        if market_price_num is not None:
            selling_price = f"${int(market_price_num * 0.8)}"
            market_price_label = f"${market_price_num:.2f}".rstrip('0').rstrip('.')

        result = {
            "label": lbl,
            "listing_id": listing_id,
            "market_price": market_price_label,
            "selling_price": selling_price,
            "description": descr if descr else "Could not fetch details.",
            "sources": clean_sources
        }

    except Exception as e:
        debug_entries.append(DebugEntry("Price+Description error", {"error": repr(e)}))
        result = {
            "label": label,
            "listing_id": listing_id,
            "market_price": "N/A",
            "selling_price": "N/A",
            "description": "Could not fetch details.",
            "sources": []
        }

    # --- 2) Category classification ---
    category = classify_category_with_llm(result["label"], debug_entries)
    result["category"] = category

    # Store session data for saving later (persist image bytes from earlier step)
    session_data = {
        'result': result,
        'images': json.loads(request.form.get('session_payload', '{}')).get('images', [])
    }
    
    return render_template_string(
        INDEX_HTML,
        result=result,
        combined=None,
        per_image=None,
        session_payload=json.dumps(session_data),
        debug_entries=debug_entries,
        categories=CATEGORIES
    )

@app.route("/save_listing", methods=["POST"])
def save_listing():
    try:
        # Get form data
        listing_id = request.form.get('listing_id')
        label = request.form.get('label')
        description = request.form.get('description')
        selling_price = request.form.get('selling_price')
        market_price = request.form.get('market_price')
        category = request.form.get('category') or "general for sale"
        
        # Get session data (images)
        session_payload = request.form.get('session_payload', '{}')
        try:
            # Handle potential double-encoding
            session_data = json.loads(session_payload)
            # If session_data is a string, parse it again
            if isinstance(session_data, str):
                session_data = json.loads(session_data)
            images_data = session_data.get('images', []) if isinstance(session_data, dict) else []
        except (json.JSONDecodeError, AttributeError) as e:
            # Fallback if JSON parsing fails
            images_data = []
        
        # Create listing directory
        listing_dir = os.path.join(LISTINGS_DIR, listing_id)
        os.makedirs(listing_dir, exist_ok=True)
        
        # Save images
        image_filenames = []
        for i, img_data in enumerate(images_data):
            try:
                if not isinstance(img_data, dict):
                    continue
                img_bytes = base64.b64decode(img_data.get('bytes', ''))
                filename = img_data.get('filename') or f"image_{i+1}.jpg"
                
                # Ensure safe filename
                safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')[:50]
                if not safe_filename:
                    safe_filename = f"image_{i+1}.jpg"
                
                img_path = os.path.join(listing_dir, safe_filename)
                with open(img_path, 'wb') as f:
                    f.write(img_bytes)
                image_filenames.append(safe_filename)
            except Exception as e:
                # Skip this image if there's an error
                continue
        
        # Create listing JSON
        listing_data = {
            'id': listing_id,
            'label': label,
            'description': description,
            'selling_price': selling_price,
            'market_price': market_price,
            'category': category,
            'images': image_filenames,
            'created_at': datetime.now().isoformat(),
            'status': 'draft'
        }
        
        # Save listing JSON
        json_path = os.path.join(listing_dir, 'listing.json')
        with open(json_path, 'w') as f:
            json.dump(listing_data, f, indent=2)
        
        # Return success
        saved_info = {
            'id': listing_id,
            'path': listing_dir,
            'files': ['listing.json'] + image_filenames
        }
        
        return render_template_string(
            INDEX_HTML,
            saved_listing=saved_info,
            debug_entries=[],
            categories=CATEGORIES
        )
        
    except Exception as e:
        return render_template_string(
            INDEX_HTML,
            error_msg=f"Error saving listing: {str(e)}",
            debug_entries=[],
            categories=CATEGORIES
        )

if __name__ == "__main__":
    # Set host="0.0.0.0" if running in a container or remote
    app.run(debug=True)
