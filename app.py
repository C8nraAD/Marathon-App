import os
import io
import json
import logging
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

try:
    from langfuse.openai import OpenAI as LfOpenAI
except ImportError:
    LfOpenAI = None

logging.basicConfig(level=logging.INFO)

DYSTANS_HM = 21.0975
MEDIANA_DEFAULT = 7200.0

st.set_page_config(page_title="Half-Marathon Predictor Pro", layout="wide")

# Profesjonalny lifting wizualny
st.markdown("""
    <style>
    .stMetric { background-color: #1e2129; padding: 15px; border-radius: 10px; border: 1px solid #3e424b; }
    [data-testid="stSidebar"] { background-color: #0e1117; }
    .msg-sep { border-bottom: 1px solid #262730; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)


AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "").strip()
MODEL_KEY = os.getenv("MODEL_BLOB_KEY", "").strip()
FILE_2023 = os.getenv("BLOB_FILE_2023", "").strip()
FILE_2024 = os.getenv("BLOB_FILE_2024", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

def format_time(sec: float) -> str:
    if pd.isna(sec) or sec <= 0: return "0:00"
    h, rem = divmod(int(round(float(sec))), 3600)
    m, s = divmod(rem, 60)
    return f"{m}:{s:02d}" if h == 0 else f"{h}:{m:02d}:{s:02d}"

def get_age_group(age: int) -> str:
    if age < 20: return "<20"
    if age < 30: return "20-29"
    if age < 40: return "30-39"
    if age < 50: return "40-49"
    if age < 60: return "50-59"
    return "60+"

# Azure Blob Storage client with caching and error handling
@st.cache_resource
def get_azure_container_client():
    if not AZURE_CONNECTION_STRING or not CONTAINER_NAME:
        logging.warning("Brak kluczy Azure. Uruchamianie w trybie offline/fallback.")
        return None
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    return blob_service_client.get_container_client(CONTAINER_NAME)

@st.cache_data
def load_benchmarks():
    container_client = get_azure_container_client()
    if not container_client:
        return {}, MEDIANA_DEFAULT
        
    def _fetch(blob_name):
        if not blob_name: return pd.DataFrame()
        try:
            blob_client = container_client.get_blob_client(blob_name)
            stream = blob_client.download_blob().readall()
            return pd.read_csv(io.BytesIO(stream), usecols=["Age", "Time_Seconds"])
        except Exception as e:
            logging.error(f"Błąd pobierania pliku {blob_name} z Azure: {e}")
            return pd.DataFrame()

    try:
        df_2023 = _fetch(FILE_2023)
        df_2024 = _fetch(FILE_2024)
        df = pd.concat([df_2023, df_2024], ignore_index=True)
        
        if df.empty:
            return {}, MEDIANA_DEFAULT
            
        df = df[(df["Age"].between(15, 90)) & df["Time_Seconds"].notna()].copy()
        df["Group"] = df["Age"].apply(get_age_group)
        bench = {g: float(grp["Time_Seconds"].median()) for g, grp in df.groupby("Group")}
        return bench, float(df["Time_Seconds"].median())
    except Exception as e: 
        logging.error(f"Krytyczny błąd przetwarzania benchmarków: {e}")
        return {}, MEDIANA_DEFAULT

BENCH_DATA, OVERALL_MEDIAN = load_benchmarks()

@st.cache_resource
def load_model():
    if not MODEL_KEY: return None
    container_client = get_azure_container_client()
    if not container_client: return None
    
    try:
        blob_client = container_client.get_blob_client(MODEL_KEY)
        stream = blob_client.download_blob().readall()
        return joblib.load(io.BytesIO(stream))
    except Exception as e: 
        logging.error(f"Błąd ładowania modelu ML z Azure: {e}")
        return None

model_ml = load_model()
client_llm = LfOpenAI() if (OPENAI_KEY and LfOpenAI) else None

with st.sidebar:
    st.header("📋 About")
    st.write("Estimates half-marathon finish times based on a short chat input. It uses basic user data to generate a pacing profile compared against 18k+ runners.")
    
    st.header("🛠️ Technology")
    st.markdown("- **Model:** `StandardScaler` → `Ridge Regression` \n- **Data:** Aggregated CSV on `Azure Blob Storage` \n- **Engine:** `Riegel's Law` optimization")
    
    st.header("📈 Performance")
    st.markdown("- **R² Score:** `0.98` \n- **MAE:** `~54 seconds` \n- **Sample:** `18,377 runners`")
    
    if st.button("Reset Application", use_container_width=True):
        st.session_state.clear()
        st.rerun()


def generate_splits(total_sec: float) -> pd.DataFrame:
    kms = [5, 10, 15, 20]
    times = [format_time(total_sec * (d/DYSTANS_HM)**1.06) for d in kms]
    return pd.DataFrame({"Distance (km)": kms, "Predicted Time": times})

if "messages" not in st.session_state: st.session_state["messages"] = []
if "answers" not in st.session_state: st.session_state["answers"] = {}
if "step" not in st.session_state: st.session_state["step"] = 0
if "pred" not in st.session_state: st.session_state["pred"] = None
if "show_balloons" not in st.session_state: st.session_state["show_balloons"] = False

def chat_flow(txt):
    if txt.lower() in ("reset", "restart"):
        st.session_state.clear()
        st.rerun()

    steps = [
        ("imie", lambda x: x.strip().title() if len(x)>1 else None, "What is your first name?"),
        ("plec", lambda x: "F" if x.lower() in ['k', 'f', 'female', 'kobieta'] else ("M" if x.lower() in ['m', 'male', 'facet'] else None), "What is your gender? (M/F)"),
        ("wiek", lambda x: int(x) if x.isdigit() and 15<=int(x)<=90 else None, "How old are you?"),
        ("tempo", lambda x: x.replace(',', ':') if (':' in x or x.isdigit()) else None, "What is your average 1 km pace? (e.g., 4:30)")
    ]
    
    idx = st.session_state.step
    if idx < len(steps):
        key, validator, q_text = steps[idx]
        val = validator(txt)
        
        if val is None: 
            return f"⚠️ Invalid input. {q_text}"
        
        st.session_state.answers[key] = val
        st.session_state.step += 1
        
        # liczmy wynik
        if st.session_state.step == 4:
            ans = st.session_state.answers
            m, s = map(int, ans["tempo"].split(':') if ':' in ans["tempo"] else (ans["tempo"], 0))
            s_5km = (m * 60 + s) * 5
            
            if model_ml:
                feats = pd.DataFrame({
                    "20_km_Czas_Sekundy": [s_5km * (20/5)**1.06],
                    "15_km_Czas_Sekundy": [s_5km * (15/5)**1.06],
                    "10_km_Czas_Sekundy": [s_5km * (10/5)**1.06],
                    "5_km_Czas_Sekundy": [s_5km],
                    "Wiek": [ans["wiek"]],
                    "Płeć_Encoded": [1 if ans["plec"] == "F" else 0]
                })
                p = float(model_ml.predict(feats)[0])
            else:
                p = s_5km * (DYSTANS_HM/5)**1.06
                
            st.session_state.pred = {"sec": p, "name": ans["imie"], "wiek": ans["wiek"]}
            st.session_state.show_balloons = True
            return f"Prediction complete! Estimated time: **{format_time(p)}**. Dashboard updated! ⮕"
        
        return steps[st.session_state.step][2]
    
    return "Analysis complete. Use the dashboard on the right."

st.title("🏃‍♂️ Half-Marathon Calculator")
col_l, col_r = st.columns([0.5, 0.5])

with col_l:
    st.subheader("💬 Assistant")
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write("Hi! Let's estimate your half-marathon time. What is your name?")
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.write(m["content"])

with col_r:
    st.subheader("📊 Performance Analysis")
    p = st.session_state.get("pred")
    
    if p:
        st.metric("⏱️ Predicted Finish Time", format_time(p["sec"]))
        
        grp = get_age_group(p["wiek"])
        m_grp = BENCH_DATA.get(grp, OVERALL_MEDIAN)
        
        df_plot = pd.DataFrame({
            "Category": ["Your Result", "Group Median", "Overall Median"],
            "Seconds": [p["sec"], m_grp, OVERALL_MEDIAN]
        })
        
        chart = alt.Chart(df_plot).mark_bar(cornerRadiusTopLeft=10).encode(
            x=alt.X("Category", sort=None, title=None),
            y=alt.Y("Seconds", title="Time (s)"),
            color=alt.Color("Category:N", 
                scale=alt.Scale(domain=["Your Result", "Group Median", "Overall Median"],
                                range=["#FF4B4B", "#4682B4", "#A9A9A9"]),
                legend=alt.Legend(title="Data Type")
            )
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### 🏃‍♂️ Projected Splits")
        st.table(generate_splits(p["sec"]))
        
        if st.button("🧠 Get AI Coach Tips", use_container_width=True):
            if client_llm:
                with st.spinner("AI Coach is thinking..."):
                    prompt_llm = f"Runner {p['name']}, age {p['wiek']}, time {format_time(p['sec'])}. Provide 3 brief tips in English."
                    res = client_llm.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt_llm}])
                    st.info(res.choices[0].message.content)
            else: st.warning("OpenAI key missing.")
        
        st.download_button("💾 Download JSON", json.dumps(p, indent=2), "result.json", use_container_width=True)
    else:
        st.info("Complete the chat on the left to unlock your racing dashboard.")


if prompt := st.chat_input("Type your answer here..."):
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Hi! Let's estimate your half-marathon time. What is your name?"})
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    bot_msg = chat_flow(prompt)
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})
    st.rerun()

if st.session_state.get("show_balloons"):
    st.balloons()
    st.session_state["show_balloons"] = False