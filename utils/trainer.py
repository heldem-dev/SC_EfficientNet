import torch
import torch.nn.functional as F


def sc_loss(p_org, p_masked, labels, criterion_cls, lambda_c=0.1):
    # Original SC objective (v1): classification on the original path,
    # plus KL consistency from the masked path to the detached original prediction.
    loss_cls = criterion_cls(p_org, labels)
    loss_cons = F.kl_div(
        F.log_softmax(p_masked, dim=1),
        F.softmax(p_org.detach(), dim=1),
        reduction="batchmean",
    )
    return loss_cls + lambda_c * loss_cons, loss_cls.detach(), loss_cons.detach()


def mask_regularizers(mask, area_target):
    """Area prior and binarization penalty for a mask of shape (B, 1, H, W).

    area:  (mean spatial mask value - area_target)^2, averaged over the batch.
           Fixes the fraction of retained positions.
    binar: mean of M * (1 - M). Pushes mask values toward 0 or 1.
    Together they rule out the near-constant mask that satisfies the v1 objective
    trivially: a constant mask equal to area_target has the maximal binarization
    penalty, so the minimum requires a spatially selective mask.
    """
    m = mask.flatten(1)
    area = ((m.mean(dim=1) - area_target) ** 2).mean()
    binar = (m * (1.0 - m)).mean()
    return area, binar


def sc_v2_loss(p_org, p_masked, mask, labels, criterion_cls, lambda_c, area_target,
               w_masked_ce, w_area, w_bin):
    # SC objective v2: both paths are supervised, the masked path is additionally
    # kept consistent with the detached original prediction, and the mask is
    # constrained to retain approximately `area_target` of the spatial positions.
    loss_org = criterion_cls(p_org, labels)
    loss_masked = criterion_cls(p_masked, labels)
    loss_cons = F.kl_div(
        F.log_softmax(p_masked, dim=1),
        F.softmax(p_org.detach(), dim=1),
        reduction="batchmean",
    )
    area, binar = mask_regularizers(mask, area_target)
    loss = (loss_org + w_masked_ce * loss_masked + lambda_c * loss_cons
            + w_area * area + w_bin * binar)
    parts = {"loss_cls": loss_org.detach(), "loss_masked_ce": loss_masked.detach(),
             "loss_cons": loss_cons.detach(), "loss_area": area.detach(), "loss_bin": binar.detach()}
    return loss, parts


def attention_only_loss(p_masked, labels, criterion_cls):
    # Ablation: remove consistency and directly supervise the inference path.
    return criterion_cls(p_masked, labels)


def train_one_epoch(
    model, loader, optimizer, device, criterion, lambda_c=0.0,
    attention_only=False, objective="sc", area_target=0.5,
    w_masked_ce=1.0, w_area=1.0, w_bin=0.1,
):
    """Returns a dict of epoch-averaged losses and mask statistics."""
    model.train()
    keys = ["loss", "loss_cls", "loss_cons", "loss_masked_ce", "loss_area", "loss_bin",
            "mask_mean", "mask_frac_below_05"]
    sums = {k: 0.0 for k in keys}

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        parts = {}

        if hasattr(model, "classifier_org"):
            p_org, p_masked, mask = model(images)
            if attention_only:
                loss = attention_only_loss(p_masked, labels, criterion)
                parts["loss_cls"] = loss.detach()
            elif objective == "sc_v2":
                loss, parts = sc_v2_loss(p_org, p_masked, mask, labels, criterion, lambda_c,
                                         area_target, w_masked_ce, w_area, w_bin)
            else:
                loss, loss_cls, loss_cons = sc_loss(p_org, p_masked, labels, criterion, lambda_c)
                parts = {"loss_cls": loss_cls, "loss_cons": loss_cons}
            with torch.no_grad():
                m = mask.flatten(1)
                parts["mask_mean"] = m.mean()
                parts["mask_frac_below_05"] = (m < 0.5).float().mean()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            parts["loss_cls"] = loss.detach()

        loss.backward()
        optimizer.step()

        bs = images.size(0)
        sums["loss"] += loss.item() * bs
        for k, v in parts.items():
            sums[k] += float(v) * bs

    n = len(loader.dataset)
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def predict(model, loader, device):
    """Returns probabilities of the inference path, labels, ids and, for SC models,
    per-image raw mask statistics (mean, fraction of positions below 0.5)."""
    model.eval()
    is_sc = hasattr(model, "classifier_org")
    if is_sc:
        model.return_mask = True
    probs, labels, ids, mstats = [], [], [], []
    try:
        for images, y, image_ids in loader:
            out = model(images.to(device))
            if is_sc:
                _, logits, mask = out
                m = mask.flatten(1)
                mstats.append(torch.stack([m.mean(1), (m < 0.5).float().mean(1),
                                           m.min(1).values, m.max(1).values], 1).cpu())
            else:
                logits = out
            probs.append(torch.softmax(logits, dim=1).cpu())
            labels.extend(y.tolist())
            ids.extend(list(image_ids))
    finally:
        if is_sc:
            model.return_mask = False
    mask_stats = torch.cat(mstats).numpy() if mstats else None
    return torch.cat(probs), labels, ids, mask_stats
