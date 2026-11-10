"""C-index (numpy), Breslow baseline hazard, and optional time-dependent AUC / Integrated Brier Score."""
import numpy as np


def concordance_index(time, event, risk):
    time, event, risk = map(np.asarray, (time, event, risk))
    num = den = 0.0
    for i in range(len(time)):
        if not event[i]:
            continue
        m = time > time[i]
        den += m.sum()
        num += (risk[i] > risk[m]).sum() + 0.5 * (risk[i] == risk[m]).sum()
    return float(num / den) if den > 0 else float("nan")


def breslow_baseline(time, event, risk):
    uniq = np.unique(time[event == 1])
    exp_r = np.exp(risk)
    cum, H = 0.0, []
    for t in uniq:
        d = ((time == t) & (event == 1)).sum()
        cum += d / exp_r[time >= t].sum()
        H.append(cum)
    return uniq, np.array(H)


def survival_probs(grid, uniq, H, risk):
    H_at = np.interp(grid, uniq, H, left=0.0)
    return np.exp(-H_at[None, :] * np.exp(risk)[:, None])


def evaluate_survival(tr_time, tr_event, tr_risk, te_time, te_event, te_risk):
    res = {"c_index": concordance_index(te_time, te_event, te_risk),
           "td_auc": float("nan"), "ibs": float("nan")}
    try:
        from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
        from sksurv.util import Surv
        y_tr = Surv.from_arrays(tr_event.astype(bool), tr_time)
        y_te = Surv.from_arrays(te_event.astype(bool), te_time)
        ev = te_time[te_event == 1]
        if len(ev) < 3:
            return res
        lo, hi = np.percentile(ev, 10), np.percentile(ev, 90)
        hi = min(hi, te_time.max() - 1e-6, tr_time.max() - 1e-6)
        grid = np.linspace(lo, hi, 15)
        res["td_auc"] = float(cumulative_dynamic_auc(y_tr, y_te, te_risk, grid)[1])
        uniq, H = breslow_baseline(tr_time, tr_event, tr_risk)
        res["ibs"] = float(integrated_brier_score(y_tr, y_te, survival_probs(grid, uniq, H, te_risk), grid))
    except Exception as e:                                                  # scikit-survival missing or too few events
        print(f"[info] td-AUC / IBS skipped: {e}")
    return res
