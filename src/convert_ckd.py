"""
Convert the UCI Chronic Kidney Disease ARFF file to a CSV.

Usage (from the project root):
    python src/convert_ckd.py data/raw/chronic_kidney_disease.arff data/raw/ckd.csv

Only parsing repairs are made here (no imputation, no encoding):
  * 2 rows have a stray trailing comma  -> removed
  * 1 row has a stray empty field       -> removed
  * '?' markers                          -> real missing values (NaN)
"""
import io
import re
import sys

import pandas as pd

NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
           "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]


def load_arff(path: str) -> pd.DataFrame:
    with open(path, encoding="latin-1") as f:
        text = f.read().replace("\r", "")

    columns = re.findall(r"@attribute\s+'?(\w+)'?\s", text, flags=re.I)
    rows = [l.strip() for l in text.split("@data", 1)[1].splitlines() if l.strip()]

    cleaned = []
    for line in rows:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(columns) + 1:
            if parts[-1] == "":
                parts = parts[:-1]                      # trailing comma
            else:
                parts = [p for p in parts if p != ""]   # stray empty field
        if len(parts) != len(columns):
            raise ValueError(f"Unexpected row length {len(parts)}: {line}")
        cleaned.append(parts)

    df = pd.DataFrame(cleaned, columns=columns).replace({"?": pd.NA, "": pd.NA})
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col])
    return df


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    df = load_arff(src)
    df.to_csv(dst, index=False)
    print(f"Saved {dst}: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df["class"].value_counts().to_dict())
