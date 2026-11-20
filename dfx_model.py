"""dfx_model.py: the XceptionNet detector for the deepfake-explanation project (Phase 2).

Lives at the REPO ROOT next to utils.py. The name is prefixed on purpose: a module called
`datasets.py` would be shadowed by the Hugging Face `datasets` package that Kaggle ships.
Heavy imports (torch, timm) stay inside functions, as in utils.py.
"""

LABEL_TO_INT = {"REAL": 0, "FAKE": 1}      # FAKE is the positive class everywhere in Phase 2
INT_TO_LABEL = {0: "REAL", 1: "FAKE"}
INPUT_SIZE = 299                            # Xception input
CANDIDATE_NAMES = ("legacy_xception", "xception")   # timm's port of the original Chollet Xception


def build_xception(pretrained=True):
    """Return (model, info). One output logit: sigmoid(logit) = P(FAKE).

    pretrained=True downloads ImageNet weights from the Hugging Face hub (needs Internet On).
    Fails loudly instead of silently training from scratch.
    """
    import timm
    names = [n for n in CANDIDATE_NAMES if timm.is_model(n)]
    if not names:
        raise RuntimeError("No original Xception in this timm version. Available: "
                           + str(timm.list_models("*xception*")))
    name = names[0]
    model = timm.create_model(name, pretrained=pretrained, num_classes=1)
    cfg = getattr(model, "pretrained_cfg", None) or {}
    info = {"timm_name": name, "timm_version": timm.__version__, "pretrained": bool(pretrained),
            "mean": tuple(cfg.get("mean", (0.5, 0.5, 0.5))), "std": tuple(cfg.get("std", (0.5, 0.5, 0.5))),
            "input_size": INPUT_SIZE}
    return model, info


def head_parameters(model):
    """Parameters of the final classifier only (stage-1 training)."""
    return list(model.get_classifier().parameters())


def set_stage(model, stage):
    """stage 1: only the head trains. stage 2: everything trains."""
    head_ids = {id(p) for p in head_parameters(model)}
    for p in model.parameters():
        p.requires_grad = True if stage == 2 else (id(p) in head_ids)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(model, path, map_location="cpu"):
    import torch
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt
