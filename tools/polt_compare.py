# tools/plot_compare.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATASET = "ETTh1"      # "ECL" or "ETTh1"
ROOT = "./dataset/"
DATE_COL = "date"
# For ECL we’ll use the 58th customer. ECL columns are usually MT_001..MT_321.
ECL_COL_INDEX = 58      # 1-based for readability; we'll convert to 0-based
ECL_COL_NAME = None     # if you know the exact name, set: "MT_058"; otherwise None to pick by index
# For ETTh1 we use the "OT" column
ETTH1_TARGET = "OT"

# Time window (change as needed)
T_START = "2012-07-15 00:00:00"  # for ECL; adjust for your data range
T_END   = "2012-07-22 23:59:59"

# Prediction paths
PRED_DYG = "./preds/dfgcn_etth1_c57.csv"   # change for ETTh1


# Optional: highlight region inside the window (tuple of start,end or None)
HIGHLIGHT = ("2012-07-18 00:00:00", "2012-07-19 23:59:59")

# ---------- HELPERS ----------
def load_ground_truth(dataset):
    if dataset == "ECL":
        f = os.path.join(ROOT, "electricity.csv")
        df = pd.read_csv(f)
        # Normalize column names
        if DATE_COL not in df.columns:
            # common ECL headers: "date" or "time"
            if "time" in df.columns: df.rename(columns={"time": DATE_COL}, inplace=True)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        # Choose customer column
        if ECL_COL_NAME and ECL_COL_NAME in df.columns:
            col = ECL_COL_NAME
        else:
            # pick by index (1-based -> 0-based). First column is date.
            # So numeric columns start at index 1.
            all_vars = [c for c in df.columns if c != DATE_COL]
            idx0 = ECL_COL_INDEX - 1
            assert 0 <= idx0 < len(all_vars), f"Bad ECL_COL_INDEX; choose 1..{len(all_vars)}"
            col = all_vars[idx0]
        return df[[DATE_COL, col]].rename(columns={col: "y"})

    elif dataset == "ETTh1":
        f = os.path.join(ROOT, "ETTh1.csv")
        df = pd.read_csv(f)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        assert ETTH1_TARGET in df.columns, f"{ETTH1_TARGET} not in ETTh1.csv"
        return df[[DATE_COL, ETTH1_TARGET]].rename(columns={ETTH1_TARGET: "y"})
    else:
        raise ValueError("dataset must be 'ECL' or 'ETTh1'")

def load_preds(path):
    # expects columns: date,pred
    df = pd.read_csv(path)
    # allow flexible headers
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date", None)
    pcol = cols.get("pred", None)
    assert dcol and pcol, f"{path} must have columns date,pred"
    df[dcol] = pd.to_datetime(df[dcol])
    return df.rename(columns={dcol: DATE_COL, pcol: "pred"})

def crop(df, start, end):
    return df[(df[DATE_COL] >= pd.to_datetime(start)) & (df[DATE_COL] <= pd.to_datetime(end))].copy()

def align(gt, p1, p2):
    m = gt.merge(p1, on=DATE_COL, how="inner").merge(p2, on=DATE_COL, how="inner", suffixes=("_dyg", "_crs"))
    # after merge, columns: date, y, pred_dyg, pred_crs
    m.rename(columns={"pred_dyg": "dyg", "pred_crs": "crs"}, inplace=True)
    return m.sort_values(DATE_COL).reset_index(drop=True)

def metrics(x, y):
    mae = float(np.mean(np.abs(x - y)))
    mse = float(np.mean((x - y) ** 2))
    return mae, mse

def peak_error(y_true, y_pred):
    # argmax within the window; difference in value and in index (timing)
    i_true = int(np.argmax(y_true))
    i_pred = int(np.argmax(y_pred))
    val_diff = float(np.abs(y_true[i_true] - y_pred[i_pred]))
    idx_diff = int(np.abs(i_true - i_pred))
    return val_diff, idx_diff, i_true, i_pred

# ---------- MAIN ----------
def main():
    gt = load_ground_truth(DATASET)
    gt_win = crop(gt, T_START, T_END)

    dyg = load_preds(PRED_DYG)
    crs = load_preds(PRED_CRS)
    dyg_win = crop(dyg, T_START, T_END)
    crs_win = crop(crs, T_START, T_END)

    df = align(gt_win, dyg_win, crs_win)

    # Metrics over window
    mae_d, mse_d = metrics(df["y"].values, df["dyg"].values)
    mae_c, mse_c = metrics(df["y"].values, df["crs"].values)

    # Peak analysis
    pk_val_d, pk_idx_d, iT, iD = peak_error(df["y"].values, df["dyg"].values)
    pk_val_c, pk_idx_c, _, iC = peak_error(df["y"].values, df["crs"].values)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df[DATE_COL], df["y"],  lw=2.0, label="Ground Truth", color="#111111")
    ax.plot(df[DATE_COL], df["dyg"], lw=2.0, label="DyGraphformer", color="#2b8cbe")
    ax.plot(df[DATE_COL], df["crs"], lw=2.0, label="Crossformer",  color="#e34a33")

    # mark peaks
    ax.scatter(df[DATE_COL].iloc[iT], df["y"].iloc[iT], s=50, color="#111111", zorder=5)
    ax.scatter(df[DATE_COL].iloc[iD], df["dyg"].iloc[iD], s=50, color="#2b8cbe", zorder=5)
    ax.scatter(df[DATE_COL].iloc[iC], df["crs"].iloc[iC], s=50, color="#e34a33", zorder=5)

    # optional highlight region
    if HIGHLIGHT:
        hs, he = pd.to_datetime(HIGHLIGHT[0]), pd.to_datetime(HIGHLIGHT[1])
        ax.axvspan(hs, he, color="#f1c40f", alpha=0.15, lw=0)

    title_left = f"{DATASET} – comparison window"
    title_right = (f"MAE/MSE DyG: {mae_d:.3f}/{mse_d:.3f} | "
                   f"MAE/MSE Cross: {mae_c:.3f}/{mse_c:.3f}\n"
                   f"Peak miss (value/ticks) DyG: {pk_val_d:.3f}/{pk_idx_d}, "
                   f"Cross: {pk_val_c:.3f}/{pk_idx_c}")
    ax.set_title(title_left + "\n" + title_right, loc="left", fontsize=11)
    ax.set_ylabel("Load")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=3, frameon=False)
    fig.tight_layout()

    out = f"./fig_compare_{DATASET.lower()}.png"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
