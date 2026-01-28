# 🧺 Market Basket Analysis – Streamlit Dashboard

This project implements an interactive **Market Basket Analysis** application
to discover product association rules and co-purchasing patterns from
transactional data.

The dashboard allows users to dynamically adjust Apriori algorithm parameters
and explore meaningful associations through tables and visualizations.

---

## 🔍 Problem Statement
The goal is to identify frequently co-purchased products using association rule
mining. Such insights can help businesses improve:
- Cross-selling strategies
- Product bundling
- Recommendation systems

---

## 🛠️ Technologies Used
- Python
- Pandas
- mlxtend (Apriori & Association Rules)
- Streamlit
- Matplotlib

---

## ✨ Key Features
- User-controlled support, confidence, and lift thresholds
- Dynamic generation of association rules
- Filtering rules by product
- Product frequency visualization
- Downloadable association rules (CSV)
- Clean and modern interactive UI

---

## 🚀 How to Run the Application

```bash
pip install -r requirements.txt
streamlit run app.py
