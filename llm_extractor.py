import json
import os


SYSTEM_PROMPT = """You extract structured patient data from free text.

The user writes simple sentences, one fact per line. You MUST map each sentence to a field.

EXTRACTION RULES:
1. Look for these EXACT phrases (case insensitive). Extract the value if found.
2. If a phrase is not found, set the field to null. Do NOT guess.
3. Return ONLY valid JSON with ALL 21 fields (use null for missing).

PHRASE MAPPING:

Age:
- "My age is X" or "I am X years old" -> age_category (1=18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44, 6=45-49, 7=50-54, 8=55-59, 9=60-64, 10=65-69, 11=70-74, 12=75-79, 13=80+)

Sex:
- "My sex is male" or "I am male" or "I am a man" -> sex: 1
- "My sex is female" or "I am female" or "I am a woman" -> sex: 0

BMI:
- "My BMI is X" -> bmi: X (float)

Smoking:
- "I smoke" or "I am a smoker" -> smoker: 1
- "I do not smoke" or "I don't smoke" or "I am not a smoker" -> smoker: 0

Blood pressure:
- "I have high blood pressure" or "I have high BP" -> high_bp: 1
- "I do not have high blood pressure" or "I don't have high blood pressure" -> high_bp: 0

Cholesterol:
- "I have high cholesterol" -> high_chol: 1
- "I do not have high cholesterol" -> high_chol: 0

Diabetes:
- "I have diabetes" -> diabetes: 1
- "I do not have diabetes" -> diabetes: 0

Stroke:
- "I have had a stroke" -> stroke: 1
- "I have not had a stroke" -> stroke: 0

Exercise:
- "I exercise regularly" or "I am physically active" -> phys_activity: 1
- "I do not exercise" or "I am not physically active" -> phys_activity: 0

Walking:
- "I have difficulty walking" -> diff_walk: 1
- "I do not have difficulty walking" -> diff_walk: 0

Alcohol:
- "I drink heavily" or "I am a heavy drinker" -> heavy_alcohol: 1
- "I do not drink heavily" -> heavy_alcohol: 0

General health:
- "My general health is excellent" -> gen_health: 1
- "My general health is very good" -> gen_health: 2
- "My general health is good" -> gen_health: 3
- "My general health is fair" -> gen_health: 4
- "My general health is poor" -> gen_health: 5

Physical health days:
- "My poor physical health days are X" -> phys_health: X (0-30)

Mental health days:
- "My poor mental health days are X" -> ment_health: X (0-30)

Education:
- "My education level is none" -> education: 1
- "My education level is elementary" -> education: 2
- "My education level is some high school" -> education: 3
- "My education level is high school grad" -> education: 4
- "My education level is some college" -> education: 5
- "My education level is college grad" -> education: 6

Income:
- "My income level is X" -> income: X (1-8)

Healthcare:
- "I have healthcare coverage" -> any_healthcare: 1
- "I do not have healthcare coverage" -> any_healthcare: 0

Cost barrier:
- "I could not see a doctor due to cost" -> no_doc_cost: 1

Cholesterol check:
- "I have had a cholesterol check in the past 5 years" -> chol_check: 1

Fruits/vegetables:
- "I eat fruits daily" -> fruits: 1
- "I eat vegetables daily" -> veggies: 1

If a sentence does not match any phrase above, ignore it.

Return JSON. Use null for any field not found.
"""


FIELD_RULES = {
    "age_category": {"type": int, "min": 1, "max": 13},
    "sex": {"type": int, "allowed": [0, 1]},
    "high_bp": {"type": int, "allowed": [0, 1]},
    "high_chol": {"type": int, "allowed": [0, 1]},
    "chol_check": {"type": int, "allowed": [0, 1]},
    "bmi": {"type": float, "min": 10.0, "max": 60.0},
    "smoker": {"type": int, "allowed": [0, 1]},
    "stroke": {"type": int, "allowed": [0, 1]},
    "diabetes": {"type": int, "allowed": [0, 1]},
    "phys_activity": {"type": int, "allowed": [0, 1]},
    "fruits": {"type": int, "allowed": [0, 1]},
    "veggies": {"type": int, "allowed": [0, 1]},
    "heavy_alcohol": {"type": int, "allowed": [0, 1]},
    "any_healthcare": {"type": int, "allowed": [0, 1]},
    "no_doc_cost": {"type": int, "allowed": [0, 1]},
    "gen_health": {"type": int, "min": 1, "max": 5},
    "ment_health": {"type": int, "min": 0, "max": 30},
    "phys_health": {"type": int, "min": 0, "max": 30},
    "diff_walk": {"type": int, "allowed": [0, 1]},
    "education": {"type": int, "min": 1, "max": 6},
    "income": {"type": int, "min": 1, "max": 8},
}


DEFAULT_VALUES = {
    "age_category": 7, "sex": 0, "high_bp": 0, "high_chol": 0,
    "chol_check": 1, "bmi": 25.0, "smoker": 0, "stroke": 0,
    "diabetes": 0, "phys_activity": 1, "fruits": 1, "veggies": 1,
    "heavy_alcohol": 0, "any_healthcare": 1, "no_doc_cost": 0,
    "gen_health": 2, "ment_health": 0, "phys_health": 0,
    "diff_walk": 0, "education": 4, "income": 5,
}


def validate_extracted_data(data):
    validated = {}
    warnings = []
    for field, rules in FIELD_RULES.items():
        value = data.get(field)
        if value is None:
            validated[field] = None
            continue
        try:
            if rules["type"] == int:
                value = int(float(value))
            elif rules["type"] == float:
                value = float(value)
        except (ValueError, TypeError):
            warnings.append("Could not parse " + field)
            validated[field] = None
            continue
        if "allowed" in rules and value not in rules["allowed"]:
            warnings.append(field + " value not allowed")
            validated[field] = None
            continue
        if "min" in rules and value < rules["min"]:
            warnings.append(field + " below minimum")
            validated[field] = None
            continue
        if "max" in rules and value > rules["max"]:
            warnings.append(field + " above maximum")
            validated[field] = None
            continue
        validated[field] = value
    return validated, warnings


def extract_gemini(user_text, api_key, max_retries=2):
    import time
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = SYSTEM_PROMPT + "\n\nUser input: " + user_text
        last_error = ""
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text), ""
            except Exception as e:
                last_error = str(e)
                if "disconnected" in last_error.lower() or "503" in last_error or "UNAVAILABLE" in last_error:
                    time.sleep(2)
                    continue
                return None, "Gemini error: " + last_error
        return None, "Gemini failed: " + last_error
    except Exception as e:
        return None, "Gemini error: " + str(e)


def extract_pollinations(user_text, api_key):
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://gen.pollinations.ai/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model="openai",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content), ""
    except json.JSONDecodeError as e:
        return None, "Pollinations JSON parse error: " + str(e)
    except Exception as e:
        return None, "Pollinations error: " + str(e)


def extract_features(user_text, provider="auto"):
    result = {
        "success": False,
        "data": None,
        "partial_data": None,
        "warnings": [],
        "error": "",
        "provider_used": "",
        "attempts": [],
        "fallback_used": False,
    }

    if provider == "auto":
        providers_to_try = ["gemini", "pollinations"]
    else:
        providers_to_try = [provider]

    for prov in providers_to_try:
        api_key = os.environ.get(prov.upper() + "_API_KEY")
        if not api_key:
            result["attempts"].append(prov + ": no API key")
            continue

        if prov == "gemini":
            raw_data, error = extract_gemini(user_text, api_key)
        elif prov == "pollinations":
            raw_data, error = extract_pollinations(user_text, api_key)
        else:
            result["attempts"].append(prov + ": unknown provider")
            continue

        if raw_data is None:
            result["attempts"].append(prov + ": " + error)
            continue

        validated, warnings = validate_extracted_data(raw_data)
        result["partial_data"] = validated
        result["warnings"] = warnings

        filled = {}
        for field in FIELD_RULES.keys():
            filled[field] = validated.get(field) if validated.get(field) is not None else DEFAULT_VALUES[field]

        result["data"] = filled
        result["success"] = True
        result["provider_used"] = prov
        result["attempts"].append(prov + ": success")

        if prov != providers_to_try[0]:
            result["fallback_used"] = True

        return result

    result["error"] = " | ".join(result["attempts"])
    result["fallback_used"] = True
    return result
