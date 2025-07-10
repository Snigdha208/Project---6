import streamlit as st
import pandas as pd
from gemini.recommender import generate_recommendations

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=['datetime'])
    return df

def spending_patterns(df):
    total_spend = df['amount'].sum()
    spend_by_category = df.groupby('merchant_category')['amount'].sum().sort_values(ascending=False)
    spend_essential = df[df['expense_type'] == 'essential']['amount'].sum()
    spend_non_essential = df[df['expense_type'] == 'non-essential']['amount'].sum()
    avg_transaction = df['amount'].mean()
    return {
        "total_spend": total_spend,
        "spend_by_category": spend_by_category,
        "spend_essential": spend_essential,
        "spend_non_essential": spend_non_essential,
        "avg_transaction": avg_transaction
    }

def build_summary_text(patterns) -> str:
    top_categories_str = patterns['spend_by_category'].head(5).to_string()
    summary = f"""
Total spend: ₹{patterns['total_spend']:.2f}
Essential spend: ₹{patterns['spend_essential']:.2f}
Non-essential spend: ₹{patterns['spend_non_essential']:.2f}
Average transaction: ₹{patterns['avg_transaction']:.2f}
Top categories:
{top_categories_str}
"""
    return summary.strip()

def validate_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    return missing

def main():
    st.title("Financial Advisor AI with Gemini LLM")

    uploaded_file = st.file_uploader("Upload your cleaned UPI transactions CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['datetime'])
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return
    else:
        st.info("Using default dataset (replace with your path or upload a file)")
        try:
            df = load_data(r'F:\vscode\UPI\data\upi_transactions_cleaned.csv')
        except Exception as e:
            st.error(f"Error loading default dataset: {e}")
            return

    required_cols = ['datetime', 'amount', 'merchant_category', 'expense_type']
    missing = validate_columns(df, required_cols)
    if missing:
        st.error(f"Missing columns in data: {missing}")
        return

    st.write("### Sample Data")
    st.dataframe(df.sample(5) if len(df) > 5 else df)

    if st.button("Analyze Spending and Generate Recommendations"):
        patterns = spending_patterns(df)
        summary_text = build_summary_text(patterns)
        st.write("### Spending Summary")
        st.code(summary_text)  # nice formatting

        with st.spinner("Generating financial advice..."):
            try:
                advice = generate_recommendations(summary_text)
                st.write("### Financial Advice from Gemini")
                st.write(advice)
            except Exception as e:
                st.error(f"Failed to generate recommendations: {e}")

if __name__ == "__main__":
    main()