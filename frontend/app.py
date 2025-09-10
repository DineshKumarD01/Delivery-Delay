import streamlit as st
import json
import datetime
import requests
import os

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Delivery Delay Prediction", layout="centered")

# Custom CSS to reduce label font size
st.markdown(
    """
    <style>
    label, .stSelectbox label, .stNumberInput label, .stDateInput label, .stTimeInput label {
        font-size: 14px !important;
    }
    .stSlider label {
        font-size: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Title
# --------------------------
st.title("🚚 Delivery Delay Prediction")
st.markdown("### Order Data Form")   # smaller subtitle
st.markdown("Fill in the details below to predict delivery delays.")

# --------------------------
# Load Hierarchy JSON Files
# --------------------------

with open("customer_hierarchy.json", "r", encoding="utf-8") as f:
    customer_hierarchy = json.load(f)

with open("order_hierarchy.json", "r", encoding="utf-8") as f:
    order_hierarchy = json.load(f)
# --------------------------
# Dropdown choices
# --------------------------
payment_types = ["CASH", "DEBIT", "PAYMENT", "TRANSFER"]
shipping_modes = ["Standard Class", "First Class", "Second Class"]

# --------------------------
# 1. Numeric Inputs with Ranges (based on stats you shared)
# --------------------------

profit_per_order = st.number_input(
    "Profit per Order",
    min_value=-4000.0, max_value=630.0, step=1.0, value=30.0,
    help="Range: -4048 to 629"
)

order_item_discount = st.number_input(
    "Order Item Discount",
    min_value=0.0, max_value=404.0, step=1.0, value=15.0,
    help="Range: 0 to 404"
)

order_item_product_price = st.number_input(
    "Product Price",
    min_value=10.0, max_value=1617.0, step=1.0, value=100.0,
    help="Range: 10 to 1617"
)

order_item_profit_ratio = st.number_input(
    "Profit Ratio",
    min_value=-2.75, max_value=0.5, step=0.01, value=0.1,
    help="Range: -2.75 to 0.5"
)

order_item_quantity = st.number_input(
    "Quantity",
    min_value=1, max_value=5, step=1, value=1,
    help="Range: 1 to 5"
)

sales = st.number_input(
    "Sales",
    min_value=10.0, max_value=1617.0, step=1.0, value=200.0,
    help="Range: 10 to 1617"
)

order_profit_per_order = st.number_input(
    "Order Profit per Order",
    min_value=-1523.0, max_value=626.0, step=1.0, value=30.0,
    help="Range: -1523 to 626"
)

# --------------------------
# 2. Customer & Order Hierarchies
# --------------------------
st.subheader("Customer Details")
customer_country = st.selectbox("Customer Country", list(customer_hierarchy.keys()))
customer_states = list(customer_hierarchy[customer_country].keys())
customer_state = st.selectbox("Customer State", customer_states)
customer_cities = customer_hierarchy[customer_country][customer_state]
customer_city = st.selectbox("Customer City", customer_cities)

st.subheader("Order Details")
order_region = st.selectbox("Order Region", list(order_hierarchy.keys()))
order_countries = list(order_hierarchy[order_region].keys())
order_country = st.selectbox("Order Country", order_countries)
order_states = list(order_hierarchy[order_region][order_country].keys())
order_state = st.selectbox("Order State", order_states)
order_cities = list(order_hierarchy[order_region][order_country][order_state].keys())
order_city = st.selectbox("Order City", order_cities)

# --------------------------
# 3. Payment & Shipping
# --------------------------
payment_type = st.selectbox("Payment Type", payment_types)
shipping_mode = st.selectbox("Shipping Mode", shipping_modes)

# --------------------------
# 4. Date + Time
# --------------------------
st.subheader("Order & Shipping Dates")

order_date = st.date_input("Order Date", datetime.date.today())
order_time = st.time_input("Order Time", datetime.datetime.now().time())

ship_date = st.date_input("Shipping Date", datetime.date.today() + datetime.timedelta(days=2))
ship_time = st.time_input("Shipping Time", (datetime.datetime.now() + datetime.timedelta(hours=2)).time())

# Merge date + time into datetime strings
order_datetime = datetime.datetime.combine(order_date, order_time)
ship_datetime = datetime.datetime.combine(ship_date, ship_time)

# Validation
if ship_datetime <= order_datetime:
    st.error("❌ Shipping datetime must be later than order datetime")
else:
    st.success("✅ Dates are valid")

# --------------------------
# Collect Data into JSON
# --------------------------
input_data = {
    "profit_per_order": profit_per_order,
    "order_item_discount": order_item_discount,
    "order_item_product_price": order_item_product_price,
    "order_item_profit_ratio": order_item_profit_ratio,
    "order_item_quantity": order_item_quantity,
    "sales": sales,
    "order_profit_per_order": order_profit_per_order,
    "customer": {
        "country": customer_country,
        "state": customer_state,
        "city": customer_city,
    },
    "order": {
        "region": order_region,
        "country": order_country,
        "state": order_state,
        "city": order_city,
    },
    "payment_type": payment_type,
    "shipping_mode": shipping_mode,
    "order_datetime": str(order_datetime),
    "ship_datetime": str(ship_datetime),
}

st.subheader("📑 Collected JSON")
st.json(input_data)

# --------------------------
# Send to Backend API
# --------------------------
backend_url = "http://65.1.95.224:8000/predict"

if st.button("Predict Delivery Delay"):
    try:
        response = requests.post(backend_url, json=input_data)
        result = response.json()

        if response.status_code == 200:
            st.success("✅ Successfully sent to API")
            # st.json(response.json())
            st.subheader("Prediction Result")
            st.write(f"**Delay Status:** {result['prediction']}")
            st.write(f"**Explanation:** {result['llm_explanation']}")

        else:
            st.error(f"❌ API Error: {response.status_code}")
    except Exception as e:
        st.error(f"⚠️ Failed to connect to API: {e}")
