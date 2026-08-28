"""Frequently-bought-together product recommendations."""

from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "feature_transactions.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "association_rules.csv"
OPERATIONAL_CODES = {"DOT", "POST", "M", "ADJUST"}


def build_baskets(data=None):
    """Create one product set per valid customer invoice."""
    if data is None:
        data = pd.read_csv(DATA_PATH, usecols=["Invoice", "StockCode", "IsCancelled", "Quantity"])
    valid = data.loc[
        ~data["IsCancelled"].astype(bool)
        & data["Quantity"].gt(0)
        & ~data["StockCode"].isin(OPERATIONAL_CODES)
    ]
    return valid.groupby("Invoice")["StockCode"].apply(lambda products: set(products)).tolist()


def mine_association_rules(data=None, min_support=0.01, min_confidence=0.2, top_n=100):
    """Mine and save Apriori rules with support, confidence, and lift."""
    baskets = build_baskets(data)
    encoder = TransactionEncoder()
    basket_matrix = pd.DataFrame.sparse.from_spmatrix(
        encoder.fit(baskets).transform(baskets, sparse=True),
        columns=encoder.columns_,
    )
    frequent = apriori(basket_matrix, min_support=min_support, use_colnames=True, low_memory=True)
    if frequent.empty:
        rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    else:
        rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
        rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
        rules = rules.sort_values(["lift", "confidence"], ascending=False).head(top_n).copy()
        rules["antecedents"] = rules["antecedents"].map(lambda values: ", ".join(sorted(values)))
        rules["consequents"] = rules["consequents"].map(lambda values: ", ".join(sorted(values)))
    rules.to_csv(OUTPUT_PATH, index=False)
    return rules


if __name__ == "__main__":
    result = mine_association_rules()
    print(result.head(10).to_string(index=False))