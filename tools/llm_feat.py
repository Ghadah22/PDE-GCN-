# tools/llm_features.py
import os, json, pandas as pd, numpy as np
from datetime import datetime
from utils.llm_client import query_llm

def add_basic_time_flags(df, ts_col="date", tz=None):
    t = pd.to_datetime(df[ts_col])
    if tz: t = t.dt.tz_localize("UTC").dt.tz_convert(tz)
    df["dow"] = t.dt.weekday
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour"] = t.dt.hour
    df["month"] = t.dt.month
    # Fourier encodings
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    return df

def llm_holiday_flags(dates_uniq, country="US", provider="openai", model="gpt-4o-mini"):
    prompt = (
      "Given these ISO dates, return JSON mapping date->"
      "{is_holiday:0/1, holiday_name:'or null'} for "+country+
      ". Dates:\n" + "\n".join(dates_uniq[:2000])  # cap prompt length if huge
    )
    text = query_llm(prompt, provider=provider, model=model)
    # be robust: try to extract JSON block
    try:
        j = json.loads(text)
    except Exception:
        # fallback: no holidays
        j = {d: {"is_holiday": 0, "holiday_name": None} for d in dates_uniq}
    return j

def merge_holidays(df, ts_col="date", holi_map=None):
    if holi_map is None:
        return df.assign(is_holiday=0, holiday_name=None)
    df["date_iso"] = pd.to_datetime(df[ts_col]).dt.strftime("%Y-%m-%d")
    df["is_holiday"] = df["date_iso"].map(lambda d: int(holi_map.get(d, {}).get("is_holiday", 0)))
    df["holiday_name"] = df["date_iso"].map(lambda d: holi_map.get(d, {}).get("holiday_name"))
    return df.drop(columns=["date_iso"])

def build_llm_covariates(csv_path, out_path, ts_col="date",
                         provider="openai", model="gpt-4o-mini",
                         country="US", tz=None):
    df = pd.read_csv(csv_path)
    df = add_basic_time_flags(df, ts_col=ts_col, tz=tz)
    dates_uniq = sorted(pd.to_datetime(df[ts_col]).dt.strftime("%Y-%m-%d").unique().tolist())
    # one-time LLM pass (cache the JSON)
    holi_map = llm_holiday_flags(dates_uniq, country=country, provider=provider, model=model)
    df = merge_holidays(df, ts_col=ts_col, holi_map=holi_map)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[llm_covariates] wrote {out_path}")
