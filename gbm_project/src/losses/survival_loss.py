import torch


def cox_ph_loss(risk, time, event):
    """
    Censoring-aware negative Cox partial log-likelihood (Breslow-style, ties handled approximately).
    risk: (B,) log-hazard; time: (B,); event: (B,) 1 = event, 0 = censored.
    """
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order].float()
    log_risk_set = torch.logcumsumexp(risk, dim=0)
    ll = (risk - log_risk_set) * event
    return -ll.sum() / event.sum().clamp(min=1.0)
