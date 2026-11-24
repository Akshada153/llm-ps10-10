"""Headline & Title Generator (PS10)
Paste an article or topic -> LLM returns 10 headlines as JSON with a
clarity score (1-10) -> shown in a table sorted by clarity.
"""
import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.8-flash"

SAMPLES = {
    "-- choose a sample --": "",
    "Sample 1: Remote work": (
        "A new survey of 2,000 employees found that people working from home "
        "three days a week report higher job satisfaction and take fewer sick "
        "days, but say they miss spontaneous conversations with colleagues."
    ),
    "Sample 2: Solar energy": (
        "India added record solar capacity this year as falling panel prices "
        "and government subsidies pushed rooftop installations in small towns."
    ),
    "Sample 3: Study tips": "How students can study better during exam season",
    "Sample 4: AI in healthcare": (
        "Hospitals are testing AI tools that read X-rays and flag possible "
        "fractures within seconds, helping doctors in busy emergency rooms."
    ),
    "Sample 5: Street food": "A food blog about the best street food in Pune",
}

PROMPT_TEMPLATE = """You are an expert editor who writes catchy, accurate headlines.

Task: Read the article or topic below and write exactly 10 different headline options.
Tone: {tone}

Rules:
- Each headline must be under 15 words and must not mislead (no false claims).
- Vary the style: question, how-to, number, bold statement, curiosity, etc.
- For each headline give a "clarity" score from 1 to 10
  (10 = instantly clear what the article is about, 1 = vague or confusing)
  and a one-line "reason" for the score.

Return ONLY a JSON list, no extra text, in this format:
[{{"headline": "...", "clarity": 8, "reason": "..."}}]

Article or topic:
\"\"\"
{text}
\"\"\"
"""


def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        st.error("GEMINI_API_KEY not found. Add it to a .env file (see .env.example).")
        st.stop()
    return genai.Client(api_key=key)


def generate_headlines(text: str, tone: str) -> list[dict]:
    client = get_client()
    prompt = PROMPT_TEMPLATE.format(text=text.strip(), tone=tone)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.9,
        ),
    )
    raw = response.text.strip()
    # Remove code fences if the model adds them
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    data = json.loads(raw)
    cleaned = []
    for item in data:
        try:
            score = int(item.get("clarity", 0))
        except (TypeError, ValueError):
            score = 0
        cleaned.append(
            {
                "Headline": str(item.get("headline", "")).strip(),
                "Clarity (1-10)": max(1, min(10, score)),
                "Reason": str(item.get("reason", "")).strip(),
            }
        )
    return cleaned


st.set_page_config(page_title="Headline & Title Generator", page_icon="📰", layout="wide")
st.title("📰 Headline & Title Generator")
st.caption("Paste an article or topic and get 10 headlines rated for clarity.")

with st.sidebar:
    st.header("Options")
    tone = st.selectbox(
        "Tone", ["Neutral news", "Catchy / viral", "Professional", "Friendly / casual"]
    )
    sample = st.selectbox("Try a sample input", list(SAMPLES.keys()))
    st.markdown("---")
    st.write(f"Model: `{MODEL}`")

default_text = SAMPLES[sample]
text = st.text_area(
    "Article or topic", value=default_text, height=220,
    placeholder="Paste your article or type a topic here...",
    key=f"text_{sample}",
)

if st.button("Generate 10 headlines", type="primary"):
    if len(text.strip()) < 10:
        st.warning("Please enter an article or topic (at least a few words).")
    else:
        with st.spinner("Writing headlines..."):
            try:
                rows = generate_headlines(text, tone)
            except json.JSONDecodeError:
                st.error("The model returned invalid JSON. Please click Generate again.")
                st.stop()
            except Exception as e:  # API errors, rate limits, etc.
                st.error(f"Something went wrong: {e}")
                st.stop()

        df = (
            pd.DataFrame(rows)
            .sort_values("Clarity (1-10)", ascending=False)
            .reset_index(drop=True)
        )
        df.index = df.index + 1
        st.subheader("Headlines (sorted by clarity)")
        st.dataframe(df, use_container_width=True)
        st.success(f"Best pick: {df.iloc[0]['Headline']}")
        st.download_button(
            "Download as CSV", df.to_csv(index=False), "headlines.csv", "text/csv"
        )
