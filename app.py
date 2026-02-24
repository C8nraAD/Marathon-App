import os
import io
import re
import json
import math
import unicodedata
import logging

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import boto3
import joblib
from dotenv import load_dotenv
from langfuse.openai import OpenAI as LfOpenAI

logging.basicConfig(level=logging.INFO)

HALF_MARATHON_DIST_KM = 21.0975
DEFAULT_STABILITY_FALLBACK = 0.045
OVERALL_MEDIAN_FALLBACK = 7200.0 

class Cols:
    AGE = "Age"
    TIME_SEC = "Time_Seconds"
    SEX_ENC = "Sex_Encoded"
    TEMPO_STAB_IMPUTED = "Pace_Stability_Imputed"

st.set_page_config(page_title="Half-Marathon Calculator", layout="wide")
st.markdown("<h1>Half-Marathon Calculator</h1>", unsafe_allow_html=True)
st.caption("Compare your predicted finish time against a dataset of 18,000+ runners.")

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
AWS_ENDPOINT_URL_S3 = os.getenv("AWS_ENDPOINT_URL_S3", "").strip()
DO_SPACE_NAME = os.getenv("DO_SPACE_NAME", "").strip()
S3_FILE_2023 = os.getenv("S3_FILE_2023", "").strip()
S3_FILE_2024 = os.getenv("S3_FILE_2024", "").strip()
MODEL_S3_KEY = os.getenv("MODEL_S3_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def seconds_to_hhmmss(sec: float) -> str:
    try:
        h, rem = divmod(int(round(float(sec))), 3600)
        m, s = divmod(rem, 60)
        return f"{m}:{s:02d}" if h == 0 else f"{h}:{m:02d}:{s:02d}"
    except Exception:
        return "Error"

def age_to_group(age: int) -> str:
    if age < 20: return "<20"
    if age < 30: return "20-29"
    if age < 40: return "30-39"
    if age < 50: return "40-49"
    if age < 60: return "50-59"
    return "60+"

def _norm(s: str) -> str:
    if not s: return ""
    return re.sub(r"\s+", " ", "".join(c for c in unicodedata.normalize("NFKD", s.strip().lower()) if not unicodedata.combining(c)))

def val_name(text: str): 
    return text.strip().split()[0] if text and len(text.strip().split()[0]) >= 2 else None

def val_gender(text: str): 
    if not text: return None
    s = _norm(text)
    if "♀" in text or s.startswith(("f", "fe", "w", "wo")): return "F"
    if "♂" in text or s.startswith(("m", "ma", "b", "bo")): return "M"
    return None

def val_age(text: str):
    m = re.search(r"(\d{2})", _norm(text)) if text else None
    return int(m.group(1)) if m and 15 <= int(m.group(1)) <= 90 else None

def val_time1k(text: str):
    if not text: return None
    s = text.strip().lower().replace(",", ":").replace("min", "m").replace("sec", "s")
    m = re.search(r"(\d{1,2}):(\d{1,2})", s)
    if m: return f"{int(m.group(1))}:{int(m.group(2)):02d}"
    if re.fullmatch(r"\d{1,2}", s): return f"{int(s)}:00"
    return None

@st.cache_resource
def s3_client():
    return boto3.client("s3", endpoint_url=AWS_ENDPOINT_URL_S3, aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

@st.cache_data
def load_data():
    s3 = s3_client()
    def _get_csv(key):
        obj = s3.get_object(Bucket=DO_SPACE_NAME, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()), usecols=lambda c: c in {Cols.AGE, Cols.TIME_SEC, Cols.TEMPO_STAB_IMPUTED})
    
    try:
        df = pd.concat([_get_csv(S3_FILE_2023), _get_csv(S3_FILE_2024)], ignore_index=True)
        df = df[(df[Cols.AGE].between(15, 90)) & df[Cols.TIME_SEC].notna()].copy()
        df["Group"] = pd.cut(df[Cols.AGE], bins=[0, 20, 30, 40, 50, 60, 999], labels=["<20", "20-29", "30-39", "40-49", "50-59", "60+"], right=False).astype(str)
        
        q = df.groupby("Group")[Cols.TIME_SEC].quantile([0.25, 0.5, 0.75]).unstack()
        bench = {g: {"median": float(q.loc[g, 0.50])} for g in q.index}
        stab = df.groupby("Group")[Cols.TEMPO_STAB_IMPUTED].median().to_dict() if Cols.TEMPO_STAB_IMPUTED in df.columns else {}
        for g in bench: stab.setdefault(g, DEFAULT_STABILITY_FALLBACK)
        
        return bench, stab, float(df[Cols.TIME_SEC].median())
    except Exception as e:
        logging.error(f"S3 Error: {e}")
        return {}, {}, OVERALL_MEDIAN_FALLBACK

AGE_BENCHMARKS, STAB_BY_GROUP, OVERALL_MEDIAN = load_data()

@st.cache_resource
def load_model():
    if not MODEL_S3_KEY: return None
    try:
        obj = s3_client().get_object(Bucket=DO_SPACE_NAME, Key=MODEL_S3_KEY)
        return joblib.load(io.BytesIO(obj["Body"].read()))
    except Exception:
        return None

model = load_model()
try:
    llm_client = LfOpenAI() if OPENAI_API_KEY else None
except Exception:
    llm_client = None

def generate_splits(total_sec: float, group: str, splits=[5, 10, 15, 20]) -> pd.DataFrame:
    km_marks = np.array(list(range(0, 22)) + [HALF_MARATHON_DIST_KM], dtype=float)
    x = km_marks / HALF_MARATHON_DIST_KM
    stability = float(STAB_BY_GROUP.get(group, DEFAULT_STABILITY_FALLBACK))
    
    amp = np.clip(0.85 + 5.5 * stability, 0.85, 1.25)
    fade = np.clip(1.015 + 5.0 * stability, 1.015, 1.10)
    shape_r = (1.0 - 0.02 * amp * np.exp(-((x - 0.12) / 0.10) ** 2) + (fade - 1.0) * np.clip((x - 0.72) / 0.28, 0, 1) ** 2)[1:]
    pace_s_per_km = (total_sec / np.sum(np.diff(km_marks) * shape_r)) * shape_r
    
    km_points = np.insert(km_marks[1:], 0, 0)
    split_times = np.diff(km_points) * pace_s_per_km
    cumulative_times = np.cumsum(split_times) 
    
    target_splits = np.interp(splits, km_marks[1:], cumulative_times)
    
    return pd.DataFrame({"Distance (km)": splits, "Predicted Time": [seconds_to_hhmmss(t) for t in target_splits]})

def explain_with_gpt(d: dict, group: str) -> str:
    if not llm_client: return "OpenAI API key missing."
    try:
        msg = f"Runner {d['name']}, age {d['age']}, target time: {seconds_to_hhmmss(d['sec'])}. Provide 3 brief half-marathon training tips."
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a running coach. Provide 3 short, actionable tips in English."}, {"role": "user", "content": msg}],
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {e}"
    
if "messages" not in st.session_state: st.session_state["messages"] = []
if "answers" not in st.session_state: st.session_state["answers"] = {"name": None, "gender": None, "age": None, "time_1km": None}
if "step_index" not in st.session_state: st.session_state["step_index"] = 0
if "last_prediction" not in st.session_state: st.session_state["last_prediction"] = None

def current_prompt():
    prompts = ["What is your first name?", "What is your gender? (M/F)", "How old are you?", "What is your average 1 km pace? (e.g., 4:30)"]
    i = st.session_state["step_index"]
    if i < len(prompts):
        return f"Hi **{st.session_state['answers']['name']}**! {prompts[i]}" if i == 1 else prompts[i]
    return "Prediction complete. Click Reset to start over."

def process_chat(user_text: str):
    if user_text.lower() in ("reset", "restart"):
        st.session_state["answers"] = {"name": None, "gender": None, "age": None, "time_1km": None}
        st.session_state["step_index"] = 0
        st.session_state["messages"].clear()
        st.session_state["last_prediction"] = None
        return current_prompt()

    steps = [
        ("name", val_name, "Invalid input. Please provide your first name."), 
        ("gender", val_gender, "Please reply with M or F."),
        ("age", val_age, "Provide your age as a number (15-90)."), 
        ("time_1km", val_time1k, "Invalid format. Provide pace like 4:30.")
    ]
    
    i = st.session_state["step_index"]
    if i < len(steps):
        key, validator, err = steps[i]
        val = validator(user_text)
        if not val: return f"{err}\n\n{current_prompt()}"
        
        st.session_state["answers"][key] = val
        st.session_state["step_index"] += 1
        
        if st.session_state["step_index"] < len(steps):
            return current_prompt()

    ans = st.session_state["answers"]
    if all(ans.values()):
        m, s = map(int, ans["time_1km"].split(":"))
        time_5km = (m * 60 + s) * 5
        sex_enc = 1 if ans["gender"] == "F" else 0
        
        if model:
            feats = pd.DataFrame({"20_km_Czas_Sekundy": [(time_5km/5)*20], "15_km_Czas_Sekundy": [(time_5km/5)*15], "10_km_Czas_Sekundy": [(time_5km/5)*10], "5_km_Czas_Sekundy": [time_5km], Cols.AGE: [ans["age"]], Cols.SEX_ENC: [sex_enc]})
            total_sec = float(model.predict(feats)[0])
        else:
            total_sec = (time_5km / 5) * HALF_MARATHON_DIST_KM

        st.session_state["last_prediction"] = {"sec": total_sec, "age": ans["age"], "name": ans["name"], "gender": ans["gender"], "time_1km": ans["time_1km"]}
        st.session_state["step_index"] += 1 
        return f"Estimated finish time: **{seconds_to_hhmmss(total_sec)}**. See the dashboard for details."

    return current_prompt()

with st.sidebar:
    st.markdown("## About")
    st.markdown(
        "Estimates half-marathon finish times based on a short chat input.\n\n"
        "It uses basic user data (Age, Gender, 1km Pace) to generate a pacing profile, "
        "which is then compared against a dataset of previous race results."
    )

    st.markdown("---")
    st.markdown("## Technology")
    st.markdown(
        "- **Model:** Scikit-learn Pipeline (StandardScaler -> Ridge Regression)\n"
        "- **UI/Chat:** Streamlit with basic RegEx validation\n"
        "- **Data:** Aggregated CSV files hosted on S3"
    )

    st.markdown("---")
    st.markdown("## Model Performance")
    st.markdown(
        "Trained and cross-validated (CV=5) on a dataset of 18,377 runners.\n\n"
        "- **R² Score: 0.98**\n"
        "- **MAE: ~54 seconds**\n\n"
        "The model uses Ridge regression to prevent overfitting, providing stable predictions "
        "that account for standard pacing variance."
    )
    st.markdown("---")

    if st.button("Reset Application", use_container_width=True):
        st.session_state["messages"].clear()
        st.session_state["answers"] = {
            "name": None, 
            "gender": None, 
            "age": None, 
            "time_1km": None
        }
        st.session_state["step_index"] = 0
        st.session_state["last_prediction"] = None
        st.rerun()

left, right = st.columns([0.55, 0.45])

with left:
    if not st.session_state["messages"]:
        st.session_state["messages"].append({"role": "assistant", "content": current_prompt()})
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Type your answer..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    resp = process_chat(prompt)
    st.session_state["messages"].append({"role": "assistant", "content": resp})
    st.rerun()

with right:
    st.subheader("Results Analysis")
    d = st.session_state.get("last_prediction")
    
    if d:
        total_sec = d["sec"]
        group = age_to_group(d["age"])
        group_median = AGE_BENCHMARKS.get(group, {}).get("median", OVERALL_MEDIAN)

        df_cmp = pd.DataFrame({"Benchmark": ["Your Time", "Overall Median", f"Group Median ({group})"], "Time (s)": [total_sec, OVERALL_MEDIAN, group_median]})
        bar = alt.Chart(df_cmp).mark_bar().encode(
            x=alt.X("Benchmark:N", sort=None), 
            y="Time (s):Q", 
            color=alt.Color("Benchmark:N", legend=None, scale=alt.Scale(range=["#A9A9A9", "#A9A9A9", "#4682B4"]))
        ).properties(height=300)
        st.altair_chart(bar, use_container_width=True)

        st.markdown("#### Projected Splits")
        st.dataframe(generate_splits(total_sec, group), hide_index=True, use_container_width=True)

        st.markdown("#### Export Data")
        json_data = json.dumps({"Name": d["name"], "Age": d["age"], "Prediction": seconds_to_hhmmss(total_sec)}, indent=2, ensure_ascii=False)
        st.download_button("Download JSON", data=json_data, file_name=f"result_{d['name']}.json", mime="application/json", use_container_width=True)

        st.markdown("#### AI Tips")
        if st.button("Get AI Running Tips", use_container_width=True):
            with st.spinner("Loading..."):
                st.info(explain_with_gpt(d, group))
    else:
        st.info("Complete the chat on the left to view the results.")