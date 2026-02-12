import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bond Scoring Dashboard", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def add_ranks(df: pd.DataFrame, score_col: str = "final_score") -> pd.DataFrame:
    # higher score = better rank (1 is best)
    df = df.copy()
    df["rank_universe"] = df[score_col].rank(ascending=False, method="min").astype(int)
    df["pctile_universe"] = (df[score_col].rank(pct=True, ascending=True) * 100).round(1)
    # pctile_universe: 0 = worst, 100 = best (depending on ascending choice); adjust if you prefer
    return df

SENSE = {
    "Delta Net Lev YoY": -1,
    "Delta Net Lev QoQ": -1,
    "GD YoY": -1,
    "GD QoQ": -1,
    "Debt/MC YoY": -1,
    "Debt/MC QoQ": -1,
    "Debt/MC": -1,
    "Delta D/E YoY": -1,
    "Delta D/E QoQ": -1,
    "Debt/Equity": -1,
    "Delta 1y E/A": +1,
    "E/A": +1,
    "5y Asset growth - Rev growth": -1,
    "Asset/Rev YoY": -1,
    "Delta EBITDA Margin YoY": +1,
    "Delta EBITDA Margin QoQ": +1,
    "Revenue - 5yr CAGR": +1,
    "Rev YoY": +1,
    "Rev QoQ": +1,
    "Delta WC/Rev 1Q": -1,
    "Delta WC/Rev 1y": -1,
    "Delta 1Q Inv days": -1,
    "Delta 1y Inv days": -1,
    "Int Cov YoY": +1,
    "Int Cov QoQ": +1,
    "Cash/STD YoY": +1,
    "Cash/STD QoQ": +1,
    "Cash/STD last Q": +1,
    "STD/TD YoY": -1,
    "STD/TD QoQ": -1,
    "STD/TD Last Q": -1,
    "Cash YoY": +1,
    "Cash QoQ": +1,
    "FCF score last 5Q": +1,
    "rank_universe":-1
}


def color_scale(s: pd.Series, sense_map: dict):
    """
    Direction-aware heatmap:
    red (bad) → orange (neutral) → green (good)
    """

    col = s.name
    direction = sense_map.get(col, 1)  # default = higher is better

    v = pd.to_numeric(s, errors="coerce")

    # Flip if lower is better
    if direction == -1:
        v = -v

    if v.notna().sum() < 2:
        return [""] * len(s)

    vmin, vmax = float(v.min()), float(v.max())
    if np.isclose(vmin, vmax):
        return [""] * len(s)

    def style(val):
        if pd.isna(val):
            return ""

        # normalize 0 → 1
        x = (val - vmin) / (vmax - vmin)

        # ----- RED → ORANGE → GREEN -----
        if x < 0.5:
            # red → orange
            r = 255
            g = int(165 * (x / 0.5))   # increase green toward orange
        else:
            # orange → green
            r = int(255 * (1 - (x - 0.5)/0.5))
            g = 165 + int((255 - 165) * ((x - 0.5)/0.5))

        return f"background-color: rgba({r},{g},0,0.35);"

    return [style(val) for val in v]


# --- UI ---
st.title("Bond Scoring Dashboard")

csv_path = st.sidebar.text_input("Path to scored universe CSV", value="scored_output_full_Latest_.csv")
id_cols_guess = ["Company Corp Ticker","Grade", 
"Delta Net Lev YoY",
        "Delta Net Lev QoQ",
        "GD YoY",
        "GD QoQ",
        "Debt/MC YoY",
        "Debt/MC QoQ",
        "Debt/MC",
        "Delta D/E YoY",
        "Delta D/E QoQ",
        "Debt/Equity",
        "Delta 1y E/A",
        "E/A",
        "5y Asset growth - Rev growth",
        "Asset/Rev YoY",
        "Delta EBITDA Margin YoY",
        "Delta EBITDA Margin QoQ",
        "Revenue - 5yr CAGR",
        "Rev YoY",
        "Rev QoQ",
        "Delta WC/Rev 1Q",
        "Delta WC/Rev 1y",
        "Delta 1Q Inv days",
        "Delta 1y Inv days",
        "Int Cov YoY",
        "Int Cov QoQ",
        "Cash/STD YoY",
        "Cash/STD QoQ",
        "Cash/STD last Q",
        "STD/TD YoY",
        "STD/TD QoQ",
        "STD/TD Last Q",
        "Cash YoY",
        "Cash QoQ",
        "FCF score last 5Q", 
"operations_score", "liquidity_score", "capital_structure", "market_score", "final_score",]

df = load_data(csv_path)

# Pick identifier col
available = df.columns.tolist()
default_id = next((c for c in id_cols_guess if c in available), available[0])
id_col = st.sidebar.selectbox("Company Corp Ticker", options=available, index=available.index(default_id))

score_col = st.sidebar.selectbox(
    "final_score",
    options=available,
    index=available.index("final_score") if "final_score" in available else 0
)

# Add ranks
df = add_ranks(df, score_col=score_col)

# Universe filters (optional)
if "Grade" in df.columns:
    rating_filter = st.sidebar.multiselect("IG/HY", sorted(df["Grade"].dropna().unique()))
    if rating_filter:
        df_view = df[df["Grade"].isin(rating_filter)].copy()
    else:
        df_view = df.copy()
else:
    df_view = df.copy()

# Search / selection
st.sidebar.markdown("### Search bonds")
query = st.sidebar.text_input("Company Corp Ticker")
if query:
    mask = df_view[id_col].astype(str).str.contains(query, case=False, na=False)
    candidates = df_view[mask].copy()
else:
    candidates = df_view.copy()

# Let user pick bonds
max_choices = min(200, len(candidates))
choices = candidates[id_col].astype(str).head(max_choices).tolist()
selected = st.sidebar.multiselect("Select bonds (from filtered list)", options=choices)

if selected:
    sel_df = df_view[df_view[id_col].astype(str).isin(selected)].copy()
else:
    st.info("Select bonds in the sidebar to view details.")
    st.stop()

# Columns to display
default_show = [id_col, score_col, "rank_universe", "pctile_universe"]
for c in ["capital_structure_score","liquidity_score","operations_score","Delta Net Lev YoY",
        "Delta Net Lev QoQ",
        "GD YoY",
        "GD QoQ",
        "Debt/MC YoY",
        "Debt/MC QoQ",
        "Debt/MC",
        "Delta D/E YoY",
        "Delta D/E QoQ",
        "Debt/Equity",
        "Delta 1y E/A",
        "E/A",
        "5y Asset growth - Rev growth",
        "Asset/Rev YoY",
        "Delta EBITDA Margin YoY",
        "Delta EBITDA Margin QoQ",
        "Revenue - 5yr CAGR",
        "Rev YoY",
        "Rev QoQ",
        "Delta WC/Rev 1Q",
        "Delta WC/Rev 1y",
        "Delta 1Q Inv days",
        "Delta 1y Inv days",
        "Int Cov YoY",
        "Int Cov QoQ",
        "Cash/STD YoY",
        "Cash/STD QoQ",
        "Cash/STD last Q",
        "STD/TD YoY",
        "STD/TD QoQ",
        "STD/TD Last Q",
        "Cash YoY",
        "Cash QoQ",
        "FCF score last 5Q"]:
    if c in sel_df.columns and c not in default_show:
        default_show.append(c)

show_cols = st.multiselect("Columns to display", options=sel_df.columns.tolist(), default=default_show)

# Show table with color scale on numeric columns
numeric_subset = [c for c in show_cols if pd.api.types.is_numeric_dtype(sel_df[c])]

styled = (sel_df[show_cols]
          .sort_values("rank_universe", ascending=True)
          .style
          .apply(lambda s: color_scale(s, SENSE), subset=numeric_subset, axis=0)
          .format(precision=3, na_rep=""))


st.subheader("Selected bonds")
st.dataframe(styled, use_container_width=True)

# Ranking context
st.subheader("Ranking context (top 20 around each selected bond)")
window = st.slider("Show +/- ranks around selected", 0, 50, 10)

for bond_id in sel_df[id_col].astype(str).tolist():
    r = int(df_view.loc[df_view[id_col].astype(str).eq(bond_id), "rank_universe"].iloc[0])
    lo, hi = max(1, r - window), r + window
    ctx = df_view[(df_view["rank_universe"] >= lo) & (df_view["rank_universe"] <= hi)] \
              .sort_values("rank_universe")[[id_col, score_col, "rank_universe"]]
    st.markdown(f"**{bond_id}** (rank {r})")
    st.dataframe(ctx, use_container_width=True)

# Download selection
out_csv = sel_df[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download selected rows as CSV", out_csv, file_name="selected_bonds.csv", mime="text/csv")
