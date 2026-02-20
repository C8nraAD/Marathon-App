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
    AGE = "Wiek"
    TIME_SEC = "Czas_Sekundy"
    SEX_ENC = "Płeć_Encoded"
    TEMPO_STAB_IMPUTED = "Tempo_Stabilnosc_Imputed"

st.set_page_config(page_title="Kalkulator Półmaratonu", layout="wide")
st.markdown("<h1>🏆 Kalkulator Półmaratonu</h1>", unsafe_allow_html=True)
st.caption("Sprawdź swój czas na tle innych biegaczy — biegniemy po sukces!")

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

# FUNKCJE POMOCNICZE
def seconds_to_hhmmss(sec: float) -> str:
    try:
        h, rem = divmod(int(round(float(sec))), 3600)
        m, s = divmod(rem, 60)
        return f"{m}:{s:02d}" if h == 0 else f"{h}:{m:02d}:{s:02d}"
    except Exception:
        return "Błąd"

def age_to_group(age: int) -> str:
    if age < 20: return "<20"
    if age < 30: return "20–29"
    if age < 40: return "30–39"
    if age < 50: return "40–49"
    if age < 60: return "50–59"
    return "60+"

def _norm(s: str) -> str:
    if not s: return ""
    return re.sub(r"\s+", " ", "".join(c for c in unicodedata.normalize("NFKD", s.strip().lower()) if not unicodedata.combining(c)))

# Walidatory danych
def val_name(text: str): 
    return text.strip().split()[0] if text and len(text.strip().split()[0]) >= 2 else None

def val_gender(text: str): 
    if not text: return None
    s = _norm(text)
    if "♀" in text or s.startswith(("k", "ko", "dz")): return "K"
    if "♂" in text or s.startswith(("m", "me", "ma", "chl", "pan")): return "M"
    return None

def val_age(text: str):
    m = re.search(r"(\d{2})", _norm(text)) if text else None
    return int(m.group(1)) if m and 15 <= int(m.group(1)) <= 90 else None

def val_time1k(text: str):
    if not text: return None
    s = text.strip().lower().replace(",", ":").replace("min", "m").replace("sek", "s")
    m = re.search(r"(\d{1,2}):(\d{1,2})", s)
    if m: return f"{int(m.group(1))}:{int(m.group(2)):02d}"
    if re.fullmatch(r"\d{1,2}", s): return f"{int(s)}:00"
    return None


# S3

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
        df["Grupa"] = pd.cut(df[Cols.AGE], bins=[0, 20, 30, 40, 50, 60, 999], labels=["<20", "20–29", "30–39", "40–49", "50–59", "60+"], right=False).astype(str)
        
        q = df.groupby("Grupa")[Cols.TIME_SEC].quantile([0.25, 0.5, 0.75]).unstack()
        bench = {g: {"median": float(q.loc[g, 0.50])} for g in q.index}
        stab = df.groupby("Grupa")[Cols.TEMPO_STAB_IMPUTED].median().to_dict() if Cols.TEMPO_STAB_IMPUTED in df.columns else {}
        for g in bench: stab.setdefault(g, DEFAULT_STABILITY_FALLBACK)
        
        return bench, stab, float(df[Cols.TIME_SEC].median())
    except Exception as e:
        logging.error(f"Błąd S3: {e}")
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

# Logika obliczeniowa

def generate_splits(total_sec: float, group: str, splits=[5, 10, 15, 20]) -> pd.DataFrame:
    km_marks = np.array(list(range(0, 22)) + [HALF_MARATHON_DIST_KM], dtype=float)
    x = km_marks / HALF_MARATHON_DIST_KM
    stability = float(STAB_BY_GROUP.get(group, DEFAULT_STABILITY_FALLBACK))
    
    amp = np.clip(0.85 + 5.5 * stability, 0.85, 1.25)
    fade = np.clip(1.015 + 5.0 * stability, 1.015, 1.10)
    shape_r = (1.0 - 0.02 * amp * np.exp(-((x - 0.12) / 0.10) ** 2) + (fade - 1.0) * np.clip((x - 0.72) / 0.28, 0, 1) ** 2)[1:]
    tempo_s_na_km = (total_sec / np.sum(np.diff(km_marks) * shape_r)) * shape_r
    
    # Wektoryzacja 
    km_points = np.insert(km_marks[1:], 0, 0)
    czasy_odcinkow = np.diff(km_points) * tempo_s_na_km
    czasy_skumulowane = np.cumsum(czasy_odcinkow) 
    
    # Interpolacja odczytuje wartości dokładnie na 5, 10, 15 i 20 km
    czasy_splitow = np.interp(splits, km_marks[1:], czasy_skumulowane)
    
    return pd.DataFrame({"Dystans (km)": splits, "Przewidywany czas": [seconds_to_hhmmss(t) for t in czasy_splitow]})

def explain_with_gpt(d: dict, grupa: str) -> str:
    if not llm_client: return "Brak klucza OpenAI API."
    try:
        msg = f"Biegacz {d['imie']}, wiek {d['wiek']}, cel: {seconds_to_hhmmss(d['sec'])}. Daj 3 krótkie rady treningowe pod półmaraton."
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Jesteś trenerem. Krótko, w punkt, po polsku."}, {"role": "user", "content": msg}],
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Błąd LLM: {e}"
    
# Session State Initialization
if "messages" not in st.session_state: st.session_state["messages"] = []
if "answers" not in st.session_state: st.session_state["answers"] = {"imie": None, "plec": None, "wiek": None, "czas_1km": None}
if "step_index" not in st.session_state: st.session_state["step_index"] = 0
if "last_prediction" not in st.session_state: st.session_state["last_prediction"] = None

def current_prompt():
    prompts = ["Jak masz na imię?", "Jaka jest Twoja płeć? (M/K)", "Ile masz lat?", "Jaki jest Twój czas na 1 km? (np. 4:30)"]
    i = st.session_state["step_index"]
    if i < len(prompts):
        return f"Miło Cię poznać, **{st.session_state['answers']['imie']}**! {prompts[i]}" if i == 1 else prompts[i]
    return "Mamy wynik! Chcesz nową prognozę? Użyj przycisku Reset w panelu bocznym."

# Czat i logika przetwarzania odpowiedzi
def process_chat(user_text: str):
    if user_text.lower() in ("reset", "restart"):
        st.session_state["answers"] = {"imie": None, "plec": None, "wiek": None, "czas_1km": None}
        st.session_state["step_index"] = 0
        st.session_state["messages"].clear()
        st.session_state["last_prediction"] = None
        return current_prompt()

    steps = [
        ("imie", val_name, "Nie rozumiem. Podaj samo imię."), 
        ("plec", val_gender, "Napisz po prostu M lub K."),
        ("wiek", val_age, "Podaj wiek liczbą (15-90)."), 
        ("czas_1km", val_time1k, "Zły format. Podaj czas np. 4:30.")
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

    # Wyliczanie 
    ans = st.session_state["answers"]
    if all(ans.values()):
        m, s = map(int, ans["czas_1km"].split(":"))
        czas_5km = (m * 60 + s) * 5
        plec_enc = 1 if ans["plec"] == "K" else 0
        
        if model:
            feats = pd.DataFrame({"20_km_Czas_Sekundy": [(czas_5km/5)*20], "15_km_Czas_Sekundy": [(czas_5km/5)*15], "10_km_Czas_Sekundy": [(czas_5km/5)*10], "5_km_Czas_Sekundy": [czas_5km], Cols.AGE: [ans["wiek"]], Cols.SEX_ENC: [plec_enc]})
            total_sec = float(model.predict(feats)[0])
        else:
            total_sec = (czas_5km / 5) * HALF_MARATHON_DIST_KM

        st.session_state["last_prediction"] = {"sec": total_sec, "wiek": ans["wiek"], "imie": ans["imie"], "plec": ans["plec"], "czas_1km": ans["czas_1km"]}
        st.session_state["step_index"] += 1 # Blokujemy dalsze pytania
        st.balloons()
        return f"Szacowany czas na półmaraton to: **{seconds_to_hhmmss(total_sec)}**. Zobacz wykresy obok!"

    return current_prompt()


with st.sidebar:
    st.markdown("## ℹ️ O aplikacji")
    st.markdown(
        "**Cel:** oszacować czas półmaratonu ze zwięzłego wywiadu w czacie.\n\n"
        "**Jak działa:** imię → płeć → wiek → czas 1 km. Następnie wyznaczamy profil tempa (22 punkty), "
        "skalowany do Twojego łącznego czasu i porównujemy go z bazą (CSV w Spaces).\n\n"
        "**Parametry:** płeć, wiek, tempo 1 km (→ czas 5 km), pochodne czasy kontrolne (10/15/20 km)."
    )

    st.markdown("---")
    st.markdown("## 🧠 Wykorzystane technologie")
    st.markdown(
        "- **Model ML:** Pipeline(StandardScaler → Ridge) (wczytywany z Spaces) — regresja czasu.\n"
        "- **Czat:** FSM + walidacja **RegEx** (deterministyczna ekstrakcja imię/płeć/wiek/czas).\n"
        "- **Profil tempa:** parametryczny, uczony z CSV (2023/2024) przez medianę stabilności tempa w grupie."
    )

    st.markdown("---")
    st.markdown("## 📐 Ocena jakości modelu")
    st.markdown(
        "**Model w aplikacji:** Pipeline(StandardScaler → Ridge Regression)  \n"
        "**Walidacja:** 5-krotna walidacja krzyżowa na połączonym zbiorze (2023 + 2024), N = **18 377** biegaczy.\n\n"
        "### 📊 Osiągi modelu (CV = 5)\n"
        "- **R² (średnie): 0.9834** → model wyjaśnia **98.34% zmienności** czasu półmaratonu.\n"
        "- **MAE (średnie): 53.95 s** → średni błąd to ok. **0.90 minuty** na pełnym dystansie.\n\n"
        "### 🎯 Co to znaczy w praktyce?\n"
        "- Ridge + skalowanie dają **stabilne i gładkie predykcje**.\n"
        "- Błąd na poziomie ~54 s jest porównywalny z szumem i niedokładnością pomiaru czasu w realnych zawodach.\n"
        "- Dlatego prognoza **±5 minut** używana w aplikacji jest konserwatywna – model statystycznie radzi sobie znacznie lepiej."
    )
    st.markdown("---")

    if st.button("🔁 Reset", use_container_width=True):
        st.session_state["messages"].clear()
        st.session_state["answers"] = {
            "imie": None, 
            "plec": None, 
            "wiek": None, 
            "czas_1km": None
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

if prompt := st.chat_input("Napisz odpowiedź…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    resp = process_chat(prompt)
    st.session_state["messages"].append({"role": "assistant", "content": resp})
    st.rerun()

with right:
    st.subheader("📈 Analiza wyników")
    d = st.session_state.get("last_prediction")
    
    if d:
        total_sec = d["sec"]
        grupa = age_to_group(d["wiek"])
        med_grupy = AGE_BENCHMARKS.get(grupa, {}).get("median", OVERALL_MEDIAN)

        # Wykres
        df_cmp = pd.DataFrame({"Pozycja": ["Twój czas", "Mediana ogółem", f"Mediana ({grupa})"], "Czas (s)": [total_sec, OVERALL_MEDIAN, med_grupy]})
        bar = alt.Chart(df_cmp).mark_bar().encode(
            x=alt.X("Pozycja:N", sort=None), 
            y="Czas (s):Q", 
            color=alt.Color("Pozycja:N", legend=None, scale=alt.Scale(range=["#A9A9A9", "#A9A9A9", "#FF6347"]))
        ).properties(height=300)
        st.altair_chart(bar, use_container_width=True)

        # Splity
        st.markdown("#### ⏱️ Przewidywane międzyczasy")
        st.dataframe(generate_splits(total_sec, grupa), hide_index=True, use_container_width=True)

        # JSON Download
        st.markdown("#### 💾 Zapis wyniku")
        json_data = json.dumps({"Imię": d["imie"], "Wiek": d["wiek"], "Prognoza": seconds_to_hhmmss(total_sec)}, indent=2, ensure_ascii=False)
        st.download_button("Pobierz JSON", data=json_data, file_name=f"wynik_{d['imie']}.json", mime="application/json", use_container_width=True)

        # GPT Explanation
        st.markdown("#### 🧠 Porada od AI")
        if st.button("Zapytaj Trenera GPT", use_container_width=True):
            with st.spinner("Generowanie porady..."):
                st.info(explain_with_gpt(d, grupa))
    else:
        st.info(" Wypełnij wywiad w czacie po lewej stronie, aby odblokować wykresy i analizę.")