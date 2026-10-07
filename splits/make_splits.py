from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "configs" / "HAM10000_metadata.csv"
OUT = ROOT / "splits" / "stratified_group_5fold.csv"
SEED = 42
N_SPLITS = 5


def main():
    df = pd.read_csv(META)
    required = {"image_id", "dx", "lesion_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"] = df["dx"].astype(str).str.upper()
    df["fold"] = -1

    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    for fold, (_, val_idx) in enumerate(
        sgkf.split(df, y=df["dx"], groups=df["lesion_id"])
    ):
        df.loc[val_idx, "fold"] = fold

    if (df["fold"] < 0).any():
        raise RuntimeError("Some samples did not receive a fold.")

    group_nfolds = df.groupby("lesion_id")["fold"].nunique()
    if (group_nfolds > 1).any():
        raise RuntimeError("Data leakage: one lesion_id appears in multiple folds.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Split saved: {OUT}")
    print(pd.crosstab(df["fold"], df["dx"]))


if __name__ == "__main__":
    main()
