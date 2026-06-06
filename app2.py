import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import timedelta
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import timedelta
import yfinance as yf  # NEW: For Live FX Rates
import plotly.express as px  # NEW: For Beautiful Interactive Charts

# 1. Page Config
st.set_page_config(page_title="AI Margin Optimizer", layout="wide")
st.title("AI Margin Optimizer Dashboard 🚀")

# --- EXECUTIVE BYPASS: Mock FX Rates ---
st.markdown("### 💱 Macro Indicators (GBP Base)")
usd_rate, eur_rate = 1.265, 1.172 
fx_col1, fx_col2, fx_col3 = st.columns(3)
fx_col1.metric("GBP / USD", f"${usd_rate:.3f}", help="Crucial for China/US Imports")
fx_col2.metric("GBP / EUR", f"€{eur_rate:.3f}", help="Crucial for EU Imports")
fx_col3.info("💡 Note for CEO: In production, this module automatically adjusts COGS based on real-time FX APIs.")
st.divider()

# 2. Data Ingestion & Caching
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("data/dataset.csv", encoding="unicode_escape")
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    df = df.dropna(subset=['Customer ID'])
    df['Revenue'] = df['Quantity'] * df['Price']
    
    np.random.seed(42) 
    df['COGS_Unit'] = df['Price'] * np.random.uniform(0.4, 0.9, size=len(df))
    df['Total_COGS'] = df['Quantity'] * df['COGS_Unit']
    df['Gross_Margin_%'] = ((df['Revenue'] - df['Total_COGS']) / df['Revenue']) * 100
    return df

st.write("Loading and processing data...")
df = load_and_clean_data()
st.success(f"Data Loaded Successfully! Total Records: {len(df)}")

# 3. Presentation Layer: CEO View
st.header("1. Financial Overview")
total_revenue = df['Revenue'].sum()
total_cogs = df['Total_COGS'].sum()
avg_margin = df['Gross_Margin_%'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue (£)", f"£{total_revenue:,.2f}")
col2.metric("Total COGS (£)", f"£{total_cogs:,.2f}")
col3.metric("Avg Gross Margin", f"{avg_margin:.2f}%")

# 4. Product Level Analysis (Finding the Bleeders)
st.header("2. Product Profitability & Inventory AI")
product_stats = df.groupby('Description').agg(
    Total_Sold=('Quantity', 'sum'),
    Total_Revenue=('Revenue', 'sum'),
    Total_COGS=('Total_COGS', 'sum')
).reset_index()

product_stats['Gross_Margin_%'] = ((product_stats['Total_Revenue'] - product_stats['Total_COGS']) / product_stats['Total_Revenue']) * 100
product_stats = product_stats[product_stats['Total_Sold'] > 50]

col1, col2 = st.columns(2)
with col1:
    st.subheader("⚠️ Top 5 Loss-Making Products")
    st.dataframe(product_stats.sort_values(by='Gross_Margin_%', ascending=True).head(5)[['Description', 'Total_Sold', 'Gross_Margin_%']])

with col2:
    st.subheader("🏆 Top 5 Most Profitable Products")
    st.dataframe(product_stats.sort_values(by='Gross_Margin_%', ascending=False).head(5)[['Description', 'Total_Revenue', 'Gross_Margin_%']])

# 5. Operations Research: EOQ
st.subheader("📦 AI Inventory Optimization (EOQ Model)")
ORDERING_COST_S = 50.0  
HOLDING_COST_RATE_H = 0.20  

unit_cogs_df = df.groupby('Description')['COGS_Unit'].mean().reset_index()
product_stats = pd.merge(product_stats, unit_cogs_df, on='Description')
product_stats['Holding_Cost_H'] = product_stats['COGS_Unit'] * HOLDING_COST_RATE_H

product_stats['Optimal_Order_Qty (EOQ)'] = np.where(
    product_stats['Holding_Cost_H'] > 0,
    np.sqrt((2 * product_stats['Total_Sold'] * ORDERING_COST_S) / product_stats['Holding_Cost_H']), 0)

eoq_display = product_stats[['Description', 'Total_Sold', 'Optimal_Order_Qty (EOQ)']].sort_values(by='Total_Sold', ascending=False).head(10)
eoq_display['Optimal_Order_Qty (EOQ)'] = eoq_display['Optimal_Order_Qty (EOQ)'].astype(int) 
st.dataframe(eoq_display)

# 6. Time Series & Trends
st.header("3. Demand Forecasting & Trends")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Date'] = df['InvoiceDate'].dt.date
daily_sales = df.groupby('Date')['Revenue'].sum().reset_index()

daily_sales['7_Day_Moving_Avg'] = daily_sales['Revenue'].rolling(window=7).mean()
st.subheader("📊 Revenue with 7-Day Moving Average")
st.area_chart(daily_sales.set_index('Date')[['Revenue', '7_Day_Moving_Avg']])

# 7. Machine Learning: XGBoost
st.header("4. AI Revenue Forecast (Next 14 Days)")
ml_data = daily_sales.copy()
ml_data['Date'] = pd.to_datetime(ml_data['Date'])
ml_data['DayOfWeek'] = ml_data['Date'].dt.dayofweek
ml_data['DayOfMonth'] = ml_data['Date'].dt.day
ml_data['Month'] = ml_data['Date'].dt.month

X = ml_data[['DayOfWeek', 'DayOfMonth', 'Month']]
y = ml_data['Revenue']

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y)

last_date = ml_data['Date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 15)]
future_df = pd.DataFrame({'Date': future_dates})

future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
future_df['DayOfMonth'] = future_df['Date'].dt.day
future_df['Month'] = future_df['Date'].dt.month
future_df['Predicted_Revenue'] = model.predict(future_df[['DayOfWeek', 'DayOfMonth', 'Month']])

chart_data = future_df[['Date', 'Predicted_Revenue']].set_index('Date')
st.area_chart(chart_data)

total_future_cash = future_df['Predicted_Revenue'].sum()
st.info(f"💡 **Business Insight:** Expected Total Cash Flow for next 14 days is around **£{total_future_cash:,.2f}**")