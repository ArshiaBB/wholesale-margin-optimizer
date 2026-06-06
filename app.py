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

# --- NEW: Live FX Tracker (Macroeconomic Context) ---
st.markdown("### 💱 Live Macro Indicators (GBP Base)")
@st.cache_data(ttl=3600) # Update once an hour so it doesn't crash
def get_fx_rates():
    try:
        # Get GBP/USD and GBP/EUR rates
        gbpusd = yf.Ticker("GBPUSD=X").history(period="1d")['Close'].iloc[-1]
        gbpeur = yf.Ticker("GBPEUR=X").history(period="1d")['Close'].iloc[-1]
        return gbpusd, gbpeur
    except:
        return None, None

usd_rate, eur_rate = get_fx_rates()

if usd_rate and eur_rate:
    fx_col1, fx_col2, fx_col3 = st.columns(3)
    fx_col1.metric("GBP / USD", f"${usd_rate:.3f}", help="Crucial for China/US Imports")
    fx_col2.metric("GBP / EUR", f"€{eur_rate:.3f}", help="Crucial for EU Imports")
    fx_col3.info("Rates are live. Crucial for COGS adjustment.")
st.divider() # Adds a nice horizontal line

# 2. Data Ingestion & Caching
@st.cache_data
def load_and_clean_data():
    # Load dataset
    df = pd.read_csv("data/dataset.csv", encoding="unicode_escape")
    
    # Clean data: Remove cancelled orders (Quantity < 0) and zero prices
    # CHANGED: 'UnitPrice' to 'Price'
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    
    # CHANGED: 'CustomerID' to 'Customer ID'
    df = df.dropna(subset=['Customer ID'])
    
    # Financial Logic: Calculate Revenue
    # CHANGED: 'UnitPrice' to 'Price'
    df['Revenue'] = df['Quantity'] * df['Price']
    
    # Financial Logic: Simulate COGS (Cost of Goods Sold) 
    np.random.seed(42) 
    # CHANGED: 'UnitPrice' to 'Price'
    df['COGS_Unit'] = df['Price'] * np.random.uniform(0.4, 0.9, size=len(df))
    df['Total_COGS'] = df['Quantity'] * df['COGS_Unit']
    
    # Calculate Gross Margin %
    df['Gross_Margin_%'] = ((df['Revenue'] - df['Total_COGS']) / df['Revenue']) * 100
    
    return df

# Load data
st.write("Loading and processing data...")
df = load_and_clean_data()

st.success(f"Data Loaded Successfully! Total Records: {len(df)}")

# 3. Presentation Layer: CEO View
st.header("1. Financial Overview")

# Calculate global metrics
total_revenue = df['Revenue'].sum()
total_cogs = df['Total_COGS'].sum()
avg_margin = df['Gross_Margin_%'].mean()

# Display KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue (£)", f"£{total_revenue:,.2f}")
col2.metric("Total COGS (£)", f"£{total_cogs:,.2f}")
col3.metric("Avg Gross Margin", f"{avg_margin:.2f}%")

st.subheader("Data Preview (Sample)")
# CHANGED: 'InvoiceNo' to 'Invoice', and 'UnitPrice' to 'Price'
st.dataframe(df[['Invoice', 'Description', 'Quantity', 'Price', 'Revenue', 'Gross_Margin_%']].head(10))

# 4. Product Level Analysis (Finding the Bleeders and Winners)
st.header("2. Product Profitability & Inventory AI")

# Group data by product
product_stats = df.groupby('Description').agg(
    Total_Sold=('Quantity', 'sum'),
    Total_Revenue=('Revenue', 'sum'),
    Total_COGS=('Total_COGS', 'sum')
).reset_index()

# Calculate Margin per product
product_stats['Gross_Margin_%'] = ((product_stats['Total_Revenue'] - product_stats['Total_COGS']) / product_stats['Total_Revenue']) * 100

# Filter out low-selling anomalies (e.g., items sold less than 50 times in total)
product_stats = product_stats[product_stats['Total_Sold'] > 50]

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ Top 5 Loss-Making Products")
    loss_makers = product_stats.sort_values(by='Gross_Margin_%', ascending=True).head(5)
    st.dataframe(loss_makers[['Description', 'Total_Sold', 'Gross_Margin_%']])

with col2:
    st.subheader("🏆 Top 5 Most Profitable Products")
    winners = product_stats.sort_values(by='Gross_Margin_%', ascending=False).head(5)
    st.dataframe(winners[['Description', 'Total_Revenue', 'Gross_Margin_%']])

# 5. Operations Research: EOQ (Economic Order Quantity) Engine
st.subheader("📦 AI Inventory Optimization (EOQ Model)")

# Business Assumptions for UK Wholesale
ORDERING_COST_S = 50.0  # £50 fixed cost per order
HOLDING_COST_RATE_H = 0.20  # 20% of item's unit COGS

# We need unit COGS for EOQ. Let's calculate an average Unit COGS per product
unit_cogs_df = df.groupby('Description')['COGS_Unit'].mean().reset_index()
product_stats = pd.merge(product_stats, unit_cogs_df, on='Description')

# EOQ Calculation
# Sqrt(2 * Demand * Ordering Cost / Holding Cost)
product_stats['Holding_Cost_H'] = product_stats['COGS_Unit'] * HOLDING_COST_RATE_H

# Calculate EOQ (handling division by zero errors)
product_stats['Optimal_Order_Qty (EOQ)'] = np.where(
    product_stats['Holding_Cost_H'] > 0,
    np.sqrt((2 * product_stats['Total_Sold'] * ORDERING_COST_S) / product_stats['Holding_Cost_H']),
    0
)

# Clean up the display dataframe
eoq_display = product_stats[['Description', 'Total_Sold', 'Optimal_Order_Qty (EOQ)']].sort_values(by='Total_Sold', ascending=False).head(10)
eoq_display['Optimal_Order_Qty (EOQ)'] = eoq_display['Optimal_Order_Qty (EOQ)'].astype(int) # Round to whole numbers

st.write("Based on Operations Research math, here is exactly how many units the purchasing manager should order at a time to minimize costs:")
st.dataframe(eoq_display)

# 6. Time Series & Demand Forecasting (Basic Trend)
st.header("3. Demand Forecasting & Trends")

# Business Logic: We need to see how revenue changes over time.
# First, convert the string date to an actual Datetime object so Python understands it.
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create a clean 'Date' column (without the hours and minutes)
df['Date'] = df['InvoiceDate'].dt.date

# Group by Date to get Daily Revenue (Like a Pivot Table by Date)
daily_sales = df.groupby('Date')['Revenue'].sum().reset_index()

# Plot the Time Series chart
# --- UPGRADED: Plotly Interactive Chart ---
st.subheader("📈 Daily Revenue Trend (Interactive)")

# Calculate 7-Day Moving Average
daily_sales['7_Day_Moving_Avg'] = daily_sales['Revenue'].rolling(window=7).mean()

# Create a beautiful Plotly chart
fig = px.line(daily_sales, x='Date', y=['Revenue', '7_Day_Moving_Avg'], 
              title='Revenue History vs 7-Day Trend',
              labels={'value': 'Amount (£)', 'variable': 'Metrics'},
              color_discrete_map={'Revenue': '#1f77b4', '7_Day_Moving_Avg': '#ff7f0e'})

fig.update_layout(hovermode="x unified", legend_title_text='') # Makes hover tooltip look pro
st.plotly_chart(fig, use_container_width=True)
# 7. Machine Learning: Predictive Analytics with XGBoost
st.header("4. AI Revenue Forecast (Next 14 Days)")

# Business Logic: Prepare data for ML
ml_data = daily_sales.copy()

# Feature Engineering: Breaking down the date into numbers ML can understand
ml_data['Date'] = pd.to_datetime(ml_data['Date'])
ml_data['DayOfWeek'] = ml_data['Date'].dt.dayofweek
ml_data['DayOfMonth'] = ml_data['Date'].dt.day
ml_data['Month'] = ml_data['Date'].dt.month

# Define Inputs (X) and Output to predict (y)
X = ml_data[['DayOfWeek', 'DayOfMonth', 'Month']]
y = ml_data['Revenue']

# Train the AI Model (Using XGBoost)
# In real life we do train/test splits, but for MVP we train on all data
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y)

# Create future dates for prediction (Next 14 Days)
last_date = ml_data['Date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 15)]
future_df = pd.DataFrame({'Date': future_dates})

# Feature Engineering for future dates
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
future_df['DayOfMonth'] = future_df['Date'].dt.day
future_df['Month'] = future_df['Date'].dt.month

# Predict!
future_df['Predicted_Revenue'] = model.predict(future_df[['DayOfWeek', 'DayOfMonth', 'Month']])

# Presentation Layer: Show the CEO the future
st.write("🔮 Based on historical patterns, here is the expected revenue for the next two weeks:")

# We format the DataFrame to look nice in the chart
chart_data = future_df[['Date', 'Predicted_Revenue']].set_index('Date')
st.line_chart(chart_data)

# Calculate total predicted cash flow
total_future_cash = future_df['Predicted_Revenue'].sum()
st.info(f"💡 **Business Insight:** Expected Total Cash Flow for next 14 days is around **£{total_future_cash:,.2f}**")