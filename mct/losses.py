"""Kayıp fonksiyonları: simetrik (CLIP-tarzı) InfoNCE + kosinüs/MSE çapası."""
import torch
import torch.nn.functional as F


def symmetric_infonce(pred, target, tau=0.07):
    """Batch-içi negatiflerle iki yönlü InfoNCE."""
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    logits = p @ t.T / tau
    labels = torch.arange(len(p), device=p.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.T, labels))


def cosine_anchor(pred, target):
    return 1 - F.cosine_similarity(pred, target, dim=-1).mean()


def mct_loss(pred, target, tau=0.07, lam_anchor=0.25, use_infonce=True):
    anchor = cosine_anchor(pred, target)
    if not use_infonce:
        return anchor, {"anchor": anchor.item()}
    nce = symmetric_infonce(pred, target, tau)
    total = nce + lam_anchor * anchor
    return total, {"infonce": nce.item(), "anchor": anchor.item()}
