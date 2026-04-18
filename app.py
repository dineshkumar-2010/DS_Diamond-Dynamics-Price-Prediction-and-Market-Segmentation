import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load models
price_model = pickle.load(open("best_random_forest_model.pkl","rb"))
cluster_model = pickle.load(open("kmeans_clustering_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
imputer = pickle.load(open("imputer.pkl","rb"))
cluster_cols = pickle.load(open("cluster_columns.pkl","rb"))

st.title("💎 Diamond Price Prediction & Market Segmentation")

st.header("Enter Diamond Details")

# Numeric Inputs
carat = st.number_input("Carat", value=0.0)
depth = st.number_input("Depth (%)", value=0.0)
table = st.number_input("Table (%)", value=0.0)
x = st.number_input("Length (x)", value=0.0)
y = st.number_input("Width (y)", value=0.0)
z = st.number_input("Height (z)", value=0.0)

# Categorical Inputs
cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"])
color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity = st.selectbox("Clarity", ["IF","VVS1","VVS2","VS1","VS2","SI1","SI2","I1"])

# Encoding (must match training)
cut_map = {"Fair":0,"Good":1,"Very Good":2,"Premium":3,"Ideal":4}
color_map = {"D":0,"E":1,"F":2,"G":3,"H":4,"I":5,"J":6}
clarity_map = {"IF":0,"VVS1":1,"VVS2":2,"VS1":3,"VS2":4,"SI1":5,"SI2":6,"I1":7}

cut = cut_map[cut]
color = color_map[color]
clarity = clarity_map[clarity]

# Feature Engineering
volume = x * y * z
length_width_ratio = x / y if y != 0 else 0
dimension_ratio = (x + y) / (2 * z) if z != 0 else 0

if carat < 0.5:
    carat_category = 0
elif carat <= 1.5:
    carat_category = 1
else:
    carat_category = 2

# Full feature set
data = pd.DataFrame([[carat, cut, color, clarity, depth, table,
                      x, y, z, volume,
                      dimension_ratio, length_width_ratio, carat_category]],
                    columns=["carat","cut","color","clarity","depth","table",
                             "x","y","z","volume",
                             "dimension_ratio","length_width_ratio","carat_category"])

# ---------------- PRICE PREDICTION ----------------
if st.button("Predict Price"):
    data_price = data.copy()
    data_price = data_price[price_model.feature_names_in_]
    price = price_model.predict(data_price)[0]
    st.success(f"💰 Predicted Price: ₹ {price:,.2f}")

# ---------------- CLUSTER PREDICTION ----------------
if st.button("Predict Market Segment"):

    # Match training columns
    data_cluster = data.copy()
    data_cluster = data_cluster.reindex(columns=cluster_cols, fill_value=0)

    # Impute + Scale
    data_imputed = imputer.transform(data_cluster)
    scaled = scaler.transform(data_imputed)

    cluster = cluster_model.predict(scaled)[0]

    cluster_names = {
        0: "Affordable Small Diamonds",
        1: "Mid-range Balanced Diamonds",
        2: "Premium Heavy Diamonds"
    }

    st.success(f"Cluster: {cluster}")
    st.info(cluster_names.get(cluster, "Unknown Segment"))
