import os
import io
import re
import json
import math
import unicodedata  
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import altair as alt
import boto3
import joblib
from dotenv import load_dotenv

#  STRONA

st.set_page_config(page_title="Kalkulator Półmaratonu", layout="wide")
st.markdown("<h1>🏆 Kalkulator Półmaratonu</h1>", unsafe_allow_html=True)
st.caption("Sprawdź swój czas na tle innych biegaczy — biegniemy po sukces!")
load_dotenv()

def _env(name: str) -> str | None:
    """Pobierz zmienną z .env i obetnij ewentualne cudzysłowy/spacje."""
    v = os.getenv(name)
    if v is None:
        return None
    return v.strip().strip("\"'")


AWS_ACCESS_KEY_ID     = _env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _env("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL_S3   = _env("AWS_ENDPOINT_URL_S3")  #
DO_SPACE_NAME         = _env("DO_SPACE_NAME")       

S3_FILE_2023 = _env("S3_FILE_2023") 
S3_FILE_2024 = _env("S3_FILE_2024")  
MODEL_S3_KEY = _env("MODEL_S3_KEY")  



def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL_S3,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


#  FUNKCJE POMOCNICZE (DEFINICJA PRZED UŻYCIEM)

def seconds_to_hhmmss(sec: float) -> str:
    try:
        sec = int(round(float(sec)))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{m}:{s:02d}" if h == 0 else f"{h}:{m:02d}:{s:02d}"
    except Exception:
        return "Błąd"

def time_to_seconds(ms: str) -> int | None:
    try:
        m, s = map(int, ms.split(":"))
        return m*60 + s
    except Exception:
        return None

def pace_label_from_total(total_sec: float) -> str:
    pace_sec = math.ceil(total_sec / 21.0975)
    mm, ss = divmod(pace_sec, 60)
    return f"{mm}:{ss:02d} / km"

def speed_kmh(total_sec: float) -> float:
    return 0.0 if total_sec <= 0 else 21.0975 / (total_sec/3600.0)

def age_to_group(age: int) -> str:
    if age < 20: return "<20"
    if age < 30: return "20–29"
    if age < 40: return "30–39"
    if age < 50: return "40–49"
    if age < 60: return "50–59"
    return "60+"

BANDS = [
    (90*60,   "Elita / 1:30↓"),
    (105*60,  "Bardzo dobry"),
    (120*60,  "Dobry (Sub-2:00)"),
    (135*60,  "Rekreacyjny+"),
    (999*60,  "Początkujący / 2:15+"),
]
def classify_result(total_sec: float) -> str:
    for thr, lab in BANDS:
        if total_sec <= thr:
            return lab
    return "—"

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"\s+"," ", s)

def parse_name(text: str) -> str|None:
    if not text:
        return None
    raw = text.strip()
    s = _norm(text)
    m = re.search(r"(nazywam sie|jestem|mam na imie)\s+([a-z\-]+)", s)
    if m:
        token = raw.split()[-1]
        return token if len(token) >= 2 else None
    t = raw.split()[0]
    return t if re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ\-]+", t) and len(t) >= 2 else None

def parse_gender(text: str) -> str|None:
    if not text:
        return None
    s = _norm(text)
    if any(ch in text for ch in ["♀","🚺"]):
        return "K"
    if any(ch in text for ch in ["♂","🚹"]):
        return "M"
    if s.startswith(("k ","ko","dz","fe","wo","pa")) or s=="k":
        return "K"
    if s.startswith(("m ","me","ma","fa","chl","pan")) or s=="m":
        return "M"
    return None

def parse_age(text: str) -> int|None:
    if not text:
        return None
    s = _norm(text)
    for pat in [r"\b(\d{1,2})\s*yo\b", r"\b(\d{1,2})\s*(lat|l)?\b", r"mam\s+(\d{1,2})", r"wiek\s+(\d{1,2})"]:
        m = re.search(pat, s)
        if m:
            age = int(m.group(1))
            return 15 <= age <= 90 and age or None
    return None

def parse_time1k(text: str) -> str|None:
    if not text:
        return None
    s = text.strip().lower().replace(" ","").replace(",",":").replace(";",":").replace("min","m").replace("sek","s")
    m = re.fullmatch(r"(\d{1,2}):(\d{1,2})", s)
    if m:
        mm, ss = map(int, m.groups())
        return f"{mm}:{ss:02d}"
    if re.fullmatch(r"\d{1,2}", s):
        return f"{int(s)}:00"
    if re.fullmatch(r"\d{3,4}", s):
        mm = int(s[:-2])
        ss = int(s[-2:])
        if 0 <= ss < 60:
            return f"{mm}:{ss:02d}"
    return None

def val_name(x): return parse_name(x)
def val_gender(x): return parse_gender(x)
def val_age(x): return parse_age(x)
def val_time1k(x):
    t = parse_time1k(x)
    if not t:
        return None
    sec = time_to_seconds(t)
    return t if sec and 150 <= sec <= 720 else None

# @st.cache_data

@st.cache_data
def load_race_data_from_spaces():
    missing = [k for k, v in {
        "AWS_ENDPOINT_URL_S3": AWS_ENDPOINT_URL_S3,
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "DO_SPACE_NAME": DO_SPACE_NAME,
        "S3_FILE_2023": S3_FILE_2023,
        "S3_FILE_2024": S3_FILE_2024,
    }.items() if not v]
    if missing:
        raise RuntimeError("Brak konfiguracji Spaces (.env): " + ", ".join(missing))

    s3 = s3_client()

    def _list_prefix(prefix: str):
        try:
            listing = s3.list_objects_v2(Bucket=DO_SPACE_NAME, Prefix=prefix)
            return [it["Key"] for it in listing.get("Contents", [])]
        except Exception:
            return []

    def _get_csv_or_explain(key: str) -> pd.DataFrame:
        try:
            obj = s3.get_object(Bucket=DO_SPACE_NAME, Key=key)
            return pd.read_csv(io.BytesIO(obj["Body"].read()))
        except s3.exceptions.NoSuchKey:
            pref = key.rsplit("/", 1)[0] + "/" if "/" in key else ""
            available = _list_prefix(pref)
            raise FileNotFoundError(
                f"❌ Nie znaleziono obiektu w Spaces\n"
                f"Bucket: {DO_SPACE_NAME}\nKey: {key}\n\n"
                f"Dostępne obiekty pod Prefix='{pref}':\n  - " + "\n  - ".join(available)
            )

    train = _get_csv_or_explain(S3_FILE_2023)
    test  = _get_csv_or_explain(S3_FILE_2024)
    df = pd.concat([train, test], ignore_index=True)

    
    df_valid = df[(df["Wiek"].between(15, 90)) & df["Czas_Sekundy"].notna()].copy()

   
    df_valid["Grupa"] = df_valid["Wiek"].astype(int).apply(age_to_group)

    
    q = df_valid.groupby("Grupa")["Czas_Sekundy"].quantile([0.25, 0.5, 0.75]).unstack()
    age_bench = {
        g: {"p25": float(q.loc[g, 0.25]), "median": float(q.loc[g, 0.50]), "p75": float(q.loc[g, 0.75])}
        for g in q.index
    }

    # mediany stabilności tempa (fallback 0.045)
    if "Tempo_Stabilnosc_Imputed" in df_valid.columns:
        stab = df_valid.groupby("Grupa")["Tempo_Stabilnosc_Imputed"].median().to_dict()
        for g in age_bench:
            stab.setdefault(g, 0.045)
    else:
        stab = {g: 0.045 for g in age_bench}

    med_by_group = {g: v["median"] for g, v in age_bench.items()}
    overall_median = float(df_valid["Czas_Sekundy"].median())

    return df, df_valid, age_bench, stab, med_by_group, overall_median

# Uruchomienie ładowania danych
RAW_ALL, POP_VALID, AGE_BENCHMARKS, STAB_BY_GROUP, MED_TIME_BY_GROUP, OVERALL_MEDIAN = load_race_data_from_spaces()

#  PROFIL TEMPA 
def data_driven_pace_profile(total_sec: float, group: str) -> pd.DataFrame:
    km_total = 21.0975
    km_marks = np.array(list(range(0, 22)) + [21.0975], dtype=float)
    x = km_marks / km_total
    stability = float(STAB_BY_GROUP.get(group, 0.045))
    
    amp   = np.clip(0.85 + 5.5*stability, 0.85, 1.25)
    fade  = np.clip(1.015 + 5.0*stability, 1.015, 1.10)
    shape = 1.0 - 0.02*amp*np.exp(-((x-0.12)/0.10)**2) + (fade-1.0)*np.clip((x-0.72)/0.28, 0, 1)**2
    
    seg_len = np.diff(km_marks)
    shape_r = shape[1:]
    C = total_sec / np.sum(seg_len * shape_r)
    pace_per_km = C * shape_r
    return pd.DataFrame({"km": km_marks[1:], "tempo_s_na_km": pace_per_km})

#  MODEL (@st.cache_resource)
@st.cache_resource
def load_model_from_spaces():
    if not MODEL_S3_KEY:
        return None
    try:
        s3 = s3_client()
        obj = s3.get_object(Bucket=DO_SPACE_NAME, Key=MODEL_S3_KEY)
        return joblib.load(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        st.warning(f"Model z Spaces niedostępny: {e}")
        return None

model = load_model_from_spaces()
MODEL_READY = model is not None

#  SESSION STATE
if "messages" not in st.session_state:
    st.session_state["messages"]=[]
if "answers"  not in st.session_state:
    st.session_state["answers"]={"name":None,"gender":None,"age":None,"time_1km_str":None}
if "step_index" not in st.session_state:
    st.session_state["step_index"]=0
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"]=None
if "show_balloons" not in st.session_state:
    st.session_state["show_balloons"] = False


#  LOGIKA CZATU (FSM)
def current_prompt():
    i = st.session_state["step_index"]
    prompts = [
        "Jak masz na imię?",
        "Jaka jest Twoja płeć? (M/K — możesz napisać też np. 'mężczyzna' / 'kobieta')",
        "Ile masz lat? (15–90 — np. '27', '27 lat', '29yo')",
        "Jaki jest Twój czas na 1 km? (np. 4:30 — wpis '5' potraktuję jako 5:00)",
    ]
    if i < len(prompts):
        if i == 1 and st.session_state["answers"]["name"]:
            return f"Miło Cię poznać, **{st.session_state['answers']['name']}**! {prompts[i]}"
        return prompts[i]
    return "Chcesz nową prognozę? Napisz „reset” lub podaj nowe imię."

def chatbot_reply(user_prompt: str, memory: list[dict]) -> dict:
    s = (user_prompt or "").strip().lower()
    if s in ("reset","zacznij od nowa","restart"):
        st.session_state["messages"].clear()
        st.session_state["answers"]={"name":None,"gender":None,"age":None,"time_1km_str":None}
        st.session_state["step_index"]=0
        st.session_state["last_prediction"]=None
        st.session_state["show_balloons"] = False
        return {"role":"assistant","content":"Zaczynamy od nowa. "+current_prompt(),"usage":{}}
    if s in ("cofnij","back"):
        st.session_state["step_index"]=max(0,st.session_state["step_index"]-1)
        return {"role":"assistant","content":"OK, wróćmy. "+current_prompt(),"usage":{}}

    steps = [
        {"key":"name","validator":val_name,"error":"Podaj imię. Np. 'Jestem Ania' albo po prostu 'Ania'."},
        {"key":"gender","validator":val_gender,"error":"Wpisz płeć: np. 'M', 'mężczyzna', 'kobieta', 'K'."},
        {"key":"age","validator":val_age,"error":"Podaj wiek (15–90). Np. '27', '27 lat', '29yo'."},
        {"key":"time_1km_str","validator":val_time1k,"error":"Podaj czas 1 km. Np. '4:30', '430', '5' (czyli 5:00)."},
    ]
    i = st.session_state["step_index"]
    if i < len(steps):
        step=steps[i]
        val=step["validator"](user_prompt)
        if val is None:
            return {"role":"assistant","content":f"{step['error']}\n\n{current_prompt()}","usage":{}}
        st.session_state["answers"][step["key"]]=val
        st.session_state["step_index"]+=1
        if st.session_state["step_index"]<len(steps):
            return {"role":"assistant","content":current_prompt(),"usage":{}}

    a = st.session_state["answers"]
    if all(a[k] is not None for k in ("name","gender","age","time_1km_str")):
        name=a["name"]; plec=a["gender"]; wiek=int(a["age"]); t1=a["time_1km_str"]
        st.session_state["answers"]={"name":None,"gender":None,"age":None,"time_1km_str":None}
        st.session_state["step_index"]=0

        try:
            czas_5km = time_to_seconds(t1)*5
            plec_enc = 1 if plec=="K" else 0
            if MODEL_READY:
                feats = pd.DataFrame({
                    "20_km_Czas_Sekundy":[(czas_5km/5)*20],
                    "15_km_Czas_Sekundy":[(czas_5km/5)*15],
                    "10_km_Czas_Sekundy":[(czas_5km/5)*10],
                    "5_km_Czas_Sekundy":[czas_5km],
                    "Wiek":[wiek],
                    "Płeć_Encoded":[plec_enc],
                })
                total_sec = float(model.predict(feats)[0])
            else:
                km_pace = czas_5km/5
                total_sec = km_pace*21.0975

            st.session_state["last_prediction"] = {
                "name": name,
                "plec": plec,
                "wiek": wiek,
                "czas_1km_str": t1,
                "czas_5km_sec_calc": int(czas_5km),
                "prediction_sec": int(round(total_sec)),
                "prediction_str": seconds_to_hhmmss(total_sec),
            }

            # ustaw flagę balonów – odpala się po rerunie, gdy karta jest już narysowana
            st.session_state["show_balloons"] = True

            # karta „Predykcja i skala” jako odpowiedź w czacie
            card_html = f"""
<div class="result-card">
  <div class="result-header">📊 Predykcja i skala</div>
  <div class="result-row ok">✅ Szacowany czas półmaratonu: <b>{seconds_to_hhmmss(total_sec)}</b> — dokładność orientacyjna <b>±5 minut</b>.</div>
  <div class="result-row band">📎 Dane: wiek <b>{wiek}</b>, płeć <b>{'K' if plec=='K' else 'M'}</b>, 1 km <b>{t1}</b></div>
</div>
"""
            return {"role":"assistant","content":card_html,"usage":{}}
        except Exception as e:
            return {"role":"assistant","content":f"Nie udało się policzyć: {e}\nSpróbuj ponownie. "+current_prompt(),"usage":{}}

    return {"role":"assistant","content":current_prompt(),"usage":{}}

#  SIDEBAR
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
        "- **Model ML:** `Pipeline(StandardScaler → Ridge)` (wczytywany z Spaces) — regresja czasu.\n"
        "- **Czat:** FSM + walidacja **RegEx** (deterministyczna ekstrakcja imię/płeć/wiek/czas).\n"
        "- **Profil tempa:** parametryczny, uczony z CSV (2023/2024) przez medianę stabilności tempa w grupie."
    )
    st.markdown("---")
    st.markdown("## 📐 Ocena jakości modelu")
    st.caption("Benchmarki liczone z 2023_TRAIN + 2024_TEST (wiek 15–90).")
    st.markdown("---")
    if st.button("🔁 Reset", use_container_width=True):
        st.session_state["messages"].clear()
        st.session_state["answers"]={"name":None,"gender":None,"age":None,"time_1km_str":None}
        st.session_state["step_index"]=0
        st.session_state["last_prediction"]=None
        st.session_state["show_balloons"] = False
        st.rerun()


#  LAYOUT (CHAT + ANALIZA)
left, right = st.columns([0.55, 0.45], vertical_alignment="top")


with left:
    if not st.session_state["messages"]:
        st.session_state["messages"].append({
            "role":"assistant",
            "content":"Cześć! Zadam kilka pytań i dam prognozę półmaratonu. "+current_prompt()
        })
    for idx, m in enumerate(st.session_state["messages"]):
        with st.chat_message(m["role"]):
            if str(m["content"]).strip().startswith("<div class=\"result-card\">"):
                st.markdown(m["content"], unsafe_allow_html=True)
            else:
                st.markdown(m["content"])
        if idx < len(st.session_state["messages"]) - 1:
            st.markdown("<div class='msg-sep'></div>", unsafe_allow_html=True)


with right:
    st.subheader("📈 Analiza porównawcza")
    d = st.session_state.get("last_prediction")
    if not d:
        st.info("Wypełnij 4 kroki w czacie — po predykcji pojawią się wykresy i analiza.")
    else:
        total_sec = float(d["prediction_sec"])
        user_group = age_to_group(int(d["wiek"])) # Ta funkcja jest już znana

       
        USER_COLOR = "#FF6347"  
        REST_COLOR = "#A9A9A9"  
        # === KONIEC ZMIANY ===

        # Wykres 1: stałe progi + Twój czas
        ref_categories = ["Twój czas", "Top 5%", "Top 15%", "Rekreacyjni", "Początkujący"]
        ref_values = [total_sec, 90*60, 105*60, 120*60, 135*60]
        # Mapowanie kolorów
        ref_colors = [USER_COLOR if cat == "Twój czas" else REST_COLOR for cat in ref_categories]

        ref = pd.DataFrame({
            "Kategoria": ref_categories,
            "Czas_s": ref_values
        })
        
        bar1 = (
            alt.Chart(ref).mark_bar().encode(
                x=alt.X("Kategoria:N", sort=None, title="Kategoria"),
                y=alt.Y("Czas_s:Q", title="Czas [s]"),
                tooltip=[alt.Tooltip("Kategoria:N"), alt.Tooltip("Czas_s:Q", format=".0f")],
                # === ZMIANA: Zastosowanie skali kolorów ===
                color=alt.Color("Kategoria:N", legend=None,
                              scale=alt.Scale(domain=ref_categories, range=ref_colors))
                # === KONIEC ZMIANY ===
            ).properties(height=360)
        )
        st.altair_chart(bar1, use_container_width=True)

        # Wykres 2: 3 kolumny — Twój czas | Mediana wszystkich | Mediana grupy
        med_all = OVERALL_MEDIAN
        med_grp = float(AGE_BENCHMARKS[user_group]["median"])
        
        cmp_categories = ["Twój czas", "Mediana wszystkich", f"Mediana {user_group}"]
        cmp_values = [total_sec, med_all, med_grp]
        # Mapowanie kolorów
        cmp_colors = [USER_COLOR if cat == "Twój czas" else REST_COLOR for cat in cmp_categories]

        df_cmp = pd.DataFrame({
            "Pozycja": cmp_categories,
            "Czas_s": cmp_values
        })
        
        bar2 = (
            alt.Chart(df_cmp).mark_bar().encode(
                x=alt.X("Pozycja:N", sort=None, title=f"Porównanie (Twoja grupa: {user_group})"),
                y=alt.Y("Czas_s:Q", title="Czas [s]"),
                tooltip=[alt.Tooltip("Pozycja:N"), alt.Tooltip("Czas_s:Q", format=".0f")],
               
                color=alt.Color("Pozycja:N", legend=None,
                              scale=alt.Scale(domain=cmp_categories, range=cmp_colors))

            ).properties(height=360)
        )
        st.altair_chart(bar2, use_container_width=True)


        # Czasy pośrednie
        st.markdown("#### ⏱️ Czasy pośrednie")
        pace_user = data_driven_pace_profile(total_sec, user_group).rename(columns={"tempo_s_na_km":"Tempo [s/km]"})
        km_points = pace_user["km"].to_numpy()
        tempo_k   = pace_user["Tempo [s/km]"].to_numpy()

        def time_at(dist_km: float) -> float:
            total = 0.0
            last = 0.0
            for i in range(len(km_points)):
                seg_end = km_points[i]
                seg_len = seg_end - last
                if dist_km >= seg_end:
                    total += tempo_k[i]*seg_len
                    last = seg_end
                else:
                    total += tempo_k[i]*(dist_km - last)
                    break
            return total

        splits = [5,10,15,20]
        times  = [seconds_to_hhmmss(time_at(km)) for km in splits]
        st.dataframe(
            pd.DataFrame({"Dystans (km)":splits,"Przewidywany czas":times}),
            use_container_width=True,
            hide_index=True
        )

        # Zapis wyniku
        st.markdown("#### 💾 Zapis wyniku")
        save_data = {
            "Imię": d["name"],
            "Płeć": d["plec"],
            "Wiek": d["wiek"],
            "Czas_1km": d["czas_1km_str"],
            "Czas_5km_s": d["czas_5km_sec_calc"],
            "Czas_polmaratonu_s": total_sec,
            "Czas_polmaratonu_HHMMSS": seconds_to_hhmmss(total_sec),
            "Srednie_tempo": pace_label_from_total(total_sec),
            "Srednia_predkosc_kmh": round(speed_kmh(total_sec),2),
            "Grupa_wiekowa": user_group,
            "Mediana_all_s": OVERALL_MEDIAN,
            "Mediana_grupy_s": med_grp,
            "Zrodlo_benchmarku": "CSV z Spaces: 2023_TRAIN + 2024_TEST (wiek 15–90)"
        }
        st.download_button(
            "Pobierz JSON",
            data=json.dumps(save_data, indent=2, ensure_ascii=False),
            file_name=f"wynik_{d['name']}.json",
            mime="application/json",
            use_container_width=True
        )

#  Chat input (fixed bottom)
prompt = st.chat_input("Napisz odpowiedź…")
if prompt:
    st.session_state["messages"].append({"role":"user","content":prompt})
    try:
        response = chatbot_reply(prompt, memory=st.session_state["messages"][-10:])
    except Exception as e:
        response = {"content": f"Wystąpił błąd: {e}", "usage": {}}
    st.session_state["messages"].append({
        "role":"assistant",
        "content":response["content"],
        "usage":response.get("usage",{})
    })
    st.rerun()

# 🎈 Balony 
if st.session_state.get("show_balloons"):
    st.balloons()
    st.session_state["show_balloons"] = False


#  Styl + JS (chat_input na dole, autoscroll, separatory)
BOTTOM_BAR_HEIGHT = 74
st.markdown(f"""
<style>
.block-container {{ padding-bottom: {BOTTOM_BAR_HEIGHT + 16}px !important; }}

/* Chat input na stałe na dole */
/* UWAGA (Senior): Poleganie na [data-testid] jest ryzykowne, */
/* może ulec zmianie w nowszych wersjach Streamlit. */
div[data-testid="stChatInput"] {{
  position: fixed !important; bottom: 0; z-index: 1000;
  left: var(--chat-left, 340px); right: var(--chat-right, 24px);
  background: var(--background-color);
  border-top: 1px solid rgba(255,255,255,0.12);
  padding-top: 4px; padding-bottom: 8px;
}}

/* Karta wyniku */
.result-card {{
  border: 2px solid rgba(255,165,0,.35);
  background: rgba(255,165,0,.06);
  border-radius: 12px; padding: 10px 14px; margin: 6px 0 4px 0;
}}
.result-header {{ font-weight: 700; margin-bottom: 6px; color: #ffb347; letter-spacing: .2px; }}
.result-row {{ border-radius: 8px; padding: 8px 10px; margin: 6px 0; font-size: 0.95rem; }}
.result-row.ok {{ background: rgba(46,204,113,.10); border: 1px solid rgba(46,204,113,.35); }}
.result-row.band {{ background: rgba(241,196,15,.10); border: 1px solid rgba(241,196,15,.35); }}

/* Subtelne linie oddzielające wiadomości/sesje czatu */
.msg-sep {{
  height: 1px;
  background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.20), rgba(255,255,255,0.06));
  margin: 8px 0 8px 0;
  border-radius: 2px;
}}
</style>
""", unsafe_allow_html=True)

components.html("""
<script>
(function () {
  function setChatWidth() {
    /* UWAGA (Senior): Poleganie na [data-testid] jest ryzykowne. */
    const sb = parent.document.querySelector('[data-testid="stSidebar"]');
    const root = parent.document.documentElement;
    const sidebarWidth = sb ? sb.offsetWidth : 0;
    root.style.setProperty('--chat-left', (sidebarWidth + 24) + 'px');
    root.style.setProperty('--chat-right', '24px');
  }
  setChatWidth();
  const sb = parent.document.querySelector('[data-testid="stSidebar"]');
  if (sb && 'ResizeObserver' in window) {
    const ro = new ResizeObserver(setChatWidth);
    ro.observe(sb);
  }
  const msgs = parent.document.querySelectorAll('[data-testid="stChatMessage"]');
  if (msgs && msgs.length) {
    msgs[msgs.length - 1].scrollIntoView({behavior: 'instant', block: 'end'});
  }
})();
</script>
""", height=0)