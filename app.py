import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from gemini.recommender import generate_recommendations

# ---------- Streamlit Page Setup ----------
st.set_page_config(page_title="AI Personal Finance Assistant", page_icon="💰", layout="wide")

# ---------- CSS Custom Styling ----------
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        font-size: 16px;
        color: #888;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #2E7D32;
    }
    .insight-box {
        background-color: #f1fdf7;
        border-left: 5px solid #2E7D32;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 8px;
        font-size: 16px;
        color: #1a1a1a;  /* <- This is the fix */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💰 AI-Powered Personal Finance Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload your cleaned UPI transactions CSV and receive smart financial advice from Google Gemini</div>', unsafe_allow_html=True)

# ---------- Data Loader ----------
@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=['datetime'])

def validate_columns(df, required_cols):
    return [col for col in required_cols if col not in df.columns]

def spending_patterns(df):
    return {
        "total_spend": df['amount'].sum(),
        "spend_by_category": df.groupby('merchant_category')['amount'].sum().sort_values(ascending=False),
        "spend_essential": df[df['expense_type'] == 'essential']['amount'].sum(),
        "spend_non_essential": df[df['expense_type'] == 'non-essential']['amount'].sum(),
        "avg_transaction": df['amount'].mean()
    }

def build_summary_text(patterns, income=None):
    top_cats = patterns['spend_by_category'].head(5)
    top_cats_str = '\n'.join([f"- {cat}: ₹{amt:.2f}" for cat, amt in top_cats.items()])
    summary = (
        f"💸 **Summary of Your Recent Spending:**\n"
        f"- Total Spend: ₹{patterns['total_spend']:.2f}\n"
        f"- Essential Spend: ₹{patterns['spend_essential']:.2f}\n"
        f"- Non-Essential Spend: ₹{patterns['spend_non_essential']:.2f}\n"
        f"- Average Transaction: ₹{patterns['avg_transaction']:.2f}\n\n"
        f"📊 **Top 5 Spending Categories:**\n{top_cats_str}"
    )

    if income:
        need = income * 0.50
        want = income * 0.30
        save = income * 0.20
        summary += (
            f"\n\n📌 **Monthly Income:** ₹{income:.2f}\n"
            f"📊 **Recommended 50/30/20 Budget Allocation:**\n"
            f"- Needs (Essentials): ₹{need:.2f}\n"
            f"- Wants (Non-Essentials): ₹{want:.2f}\n"
            f"- Savings: ₹{save:.2f}"
        )

    return summary

def plot_category_spend(df):
    data = df.groupby('merchant_category')['amount'].sum().reset_index()
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('merchant_category:N', sort='-y', title="Category"),
        y=alt.Y('amount:Q', title='Total Spend'),
        tooltip=['merchant_category', 'amount']
    ).properties(width=600, height=300, title="Category-wise Spending")
    return chart

def plot_expense_type_distribution(df):
    pie_data = df.groupby('expense_type')['amount'].sum().reset_index()
    pie = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta(field='amount', type='quantitative'),
        color=alt.Color(field='expense_type', type='nominal'),
        tooltip=['expense_type', 'amount']
    ).properties(width=400, height=300, title="Essential vs Non-Essential Spend")
    return pie

# ---------- Main App ----------
def main():
    uploaded_file = st.file_uploader("📂 Upload your cleaned UPI CSV", type=["csv"])
    monthly_income = st.number_input("💼 Enter your Monthly Income (optional)", min_value=0, step=1000)

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['datetime'])
            st.success("✅ File uploaded and loaded successfully!")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return
    else:
        st.info("No file uploaded. Using sample data.")
        try:
            df = load_data("F:/vscode/UPI/data/upi_transactions_cleaned.csv")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            return

    required_cols = ['datetime', 'amount', 'merchant_category', 'expense_type']
    missing = validate_columns(df, required_cols)
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    st.markdown("### 🧾 Sample Transactions")
    st.dataframe(df.head(5))

    if st.button("🔍 Analyze Spending and Generate Advice"):
        df_small = df.head(3)  # prevent free-tier exhaustion
        patterns = spending_patterns(df_small)
        summary_text = build_summary_text(patterns, income=monthly_income if monthly_income > 0 else None)
        
        # 💬 Rule-based personalized tip
        if patterns["spend_non_essential"] > 0.5 * patterns["total_spend"]:
            st.warning("💡 Over 50% of your expenses are non-essential. Consider reducing impulse purchases.")

        st.markdown('<div class="section-header">📋 Spending Summary</div>', unsafe_allow_html=True)
        st.markdown(summary_text)

        st.markdown('<div class="section-header">📊 Visual Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(plot_category_spend(df), use_container_width=True)
        with col2:
            st.altair_chart(plot_expense_type_distribution(df), use_container_width=True)

        st.markdown('<div class="section-header">💡 AI Financial Advice</div>', unsafe_allow_html=True)
        with st.spinner("💭 Generating advice from Gemini..."):
            if 'advice' not in st.session_state:
                st.session_state.advice = generate_recommendations(summary_text)

        if "limit" in st.session_state.advice.lower():
            st.warning(st.session_state.advice)
        else:
            st.markdown(f"<div class='insight-box'>{st.session_state.advice}</div>", unsafe_allow_html=True)
            st.balloons()

if __name__ == "__main__":
    main()