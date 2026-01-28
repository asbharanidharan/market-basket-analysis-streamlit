import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Market Basket Analysis Dashboard",
    layout="wide"
)

# ======================================================
# GLASSMORPHISM / MODERN BUSINESS THEME
# ======================================================
st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
    font-family: "Inter", "Segoe UI", sans-serif;
    color: #1f2937;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(12px);
    border-right: 1px solid #e5e7eb;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h2 {
    color: #1e3a8a;
    font-weight: 600;
}

/* Titles */
h1 {
    color: #1e3a8a;
    font-weight: 800;
}

h2, h3 {
    color: #334155;
    margin-top: 28px;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
}

/* Tables */
thead tr th {
    background-color: #e0e7ff !important;
    color: #1e3a8a !important;
    font-weight: 700;
}

tbody tr td {
    background-color: rgba(255, 255, 255, 0.65) !important;
    color: #1f2937 !important;
}

/* Alerts */
div.stAlert {
    background: rgba(255, 255, 255, 0.7);
    border-left: 5px solid #6366f1;
    border-radius: 12px;
}

/* Buttons */
button {
    background: linear-gradient(135deg, #6366f1, #3b82f6) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 8px 14px !important;
}

/* Dropdowns */
div[data-baseweb="select"] > div {
    background-color: rgba(255, 255, 255, 0.9);
}

/* Sliders */
div[data-baseweb="slider"] {
    color: #4f46e5;
}

</style>
""", unsafe_allow_html=True)

plt.style.use("default")

# ======================================================
# TITLE
# ======================================================
st.markdown("# 🧺 Market Basket Analysis Dashboard")
st.markdown(
    "A modern interactive dashboard to discover **product associations** "
    "and **co-purchasing patterns** using the Apriori algorithm."
)

st.markdown("---")

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/baskets.csv")

df = load_data()
df["products"] = df["products"].str.split(",")

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
st.sidebar.markdown("## ⚙️ User Controls")

max_products = st.sidebar.slider(
    "Number of Products to Analyze",
    20, 200, 30, 10
)

min_support = st.sidebar.slider(
    "Minimum Support",
    0.001, 0.05, 0.01, 0.001
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    0.05, 0.6, 0.15, 0.05
)

min_lift = st.sidebar.slider(
    "Minimum Lift",
    0.5, 3.0, 0.8, 0.1
)

top_n_rules = st.sidebar.slider(
    "Number of Rules to Display",
    5, 30, 10
)

# ======================================================
# PREPROCESS (MEMORY SAFE)
# ======================================================
product_counts = (
    df.explode("products")["products"]
    .value_counts()
    .head(max_products)
)

top_products = set(product_counts.index)

df["filtered_products"] = df["products"].apply(
    lambda x: [p for p in x if p in top_products]
)

transactions = df["filtered_products"].tolist()

# ======================================================
# TRANSACTION ENCODING
# ======================================================
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions, sparse=True)
basket_df = pd.DataFrame.sparse.from_spmatrix(
    te_array,
    columns=te.columns_
)

# ======================================================
# APRIORI + ASSOCIATION RULES
# ======================================================
frequent_itemsets = apriori(
    basket_df,
    min_support=min_support,
    use_colnames=True,
    low_memory=True
)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=min_confidence
)

rules = rules[rules["lift"] >= min_lift]
rules = rules.sort_values("lift", ascending=False)

# ======================================================
# METRICS
# ======================================================
st.markdown("## 📌 Key Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Total Transactions", len(df))
c2.metric("Products Analyzed", len(top_products))
c3.metric("Association Rules Found", len(rules))

st.markdown("---")

# ======================================================
# FILTER BY PRODUCT
# ======================================================
st.markdown("## 🔍 Filter Rules by Product")

product_filter = st.selectbox(
    "Select a product (optional)",
    ["All"] + sorted(top_products)
)

filtered_rules = rules.copy()
if product_filter != "All":
    filtered_rules = filtered_rules[
        filtered_rules["antecedents"].apply(lambda x: product_filter in x)
        | filtered_rules["consequents"].apply(lambda x: product_filter in x)
    ]

# ======================================================
# DISPLAY RULES
# ======================================================
st.markdown("## 📋 Top Association Rules")

if filtered_rules.empty:
    st.warning(
        "No association rules found with the current thresholds. "
        "Try lowering support, confidence, or lift."
    )
else:
    display_rules = filtered_rules.head(top_n_rules).copy()
    display_rules["antecedents"] = display_rules["antecedents"].apply(lambda x: ", ".join(x))
    display_rules["consequents"] = display_rules["consequents"].apply(lambda x: ", ".join(x))

    st.dataframe(
        display_rules[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ],
        use_container_width=True
    )

# ======================================================
# PRODUCT FREQUENCY CHART
# ======================================================
st.markdown("## 📊 Top Products by Frequency")

fig, ax = plt.subplots(figsize=(10, 4))
product_counts.head(10).plot(kind="bar", ax=ax)
ax.set_ylabel("Frequency")
ax.set_xlabel("Product")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# ======================================================
# DOWNLOAD
# ======================================================
st.markdown("## 📥 Download Results")

if not filtered_rules.empty:
    csv = display_rules.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Association Rules (CSV)",
        csv,
        "association_rules.csv",
        "text/csv"
    )

# ======================================================
# INSIGHTS
# ======================================================
st.markdown("## 📌 Insights")
st.markdown("""
- **Lift > 1** indicates strong co-purchasing behavior  
- **Confidence** reflects the reliability of a rule  
- Highly frequent products often act as **association hubs**  
- Adjusting thresholds helps explore **strong vs weak patterns**  

This dashboard demonstrates an **end-to-end Market Basket Analysis workflow**
with **user-driven exploration and professional visualization**.
""")
