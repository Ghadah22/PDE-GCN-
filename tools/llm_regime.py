# tools/llm_regime.py
import json, numpy as np, pandas as pd
from utils.llm_client import query_llm

REGIME_INS = (
 "You are a labeling bot. Using ONLY the numbers, label regime per window: "
 "one of ['trend_up','trend_down','flat','volatile','holiday_distortion','heatwave','post_event_rebound'].\n"
 "Return JSON list matching windows: [{regime:..., strength:0..1}]"
)

def windows(arr, win=168, step=24):
    for s in range(0, len(arr)-win+1, step):
        yield s, arr[s:s+win]

def label_regimes(y, win=168, step=24, provider="openai", model="gpt-4o-mini"):
    feats = []
    slices = list(windows(y, win, step))
    for s, w in slices:
        d = {
          "slope": float(np.polyfit(np.arange(len(w)), w, 1)[0]),
          "vol": float(np.std(w)),
          "last_z": float((w[-1]-np.mean(w))/(np.std(w)+1e-6)),
          "acf1": float(np.corrcoef(w[:-1], w[1:])[0,1])
        }
        feats.append(d)
    prompt = "FEATURES:\n"+json.dumps(feats)
    out = query_llm(prompt, provider=provider, model=model, system_msg=REGIME_INS)
    try:
        labels = json.loads(out)
    except Exception:
        labels = [{"regime":"flat","strength":0.5} for _ in feats]
    # expand back to full length by repeating each label over its window
    lab = np.empty(len(y), dtype=object); lab[:] = None
    strg = np.zeros(len(y))
    for i,(s,w) in enumerate(slices):
        lab[s:s+win] = labels[i]["regime"]
        strg[s:s+win] = labels[i].get("strength", 0.5)
    return lab, strg
