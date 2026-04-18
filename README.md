💎 Diamond Price Prediction & Clustering
📌 Project

This project predicts the price of a diamond and classifies it into a market segment using Machine Learning.

🚀 Features
💰 Predict diamond price (in INR)
📊 Classify diamonds into:
Affordable Small Diamonds
Mid-range Balanced Diamonds
Premium Heavy Diamonds
🧠 Models Used
Random Forest Regressor → Price Prediction
KMeans Clustering → Market Segmentation
📊 Input Features
Carat
Cut
Color
Clarity
Depth
Table
x, y, z (dimensions)
🏗 Feature Engineering
Volume = x × y × z
Dimension Ratio
Length-Width Ratio
Carat Category
📂 Files
app.py
best_random_forest_model.pkl
kmeans_clustering_model.pkl
scaler.pkl
imputer.pkl
cluster_columns.pkl
⚙️ How to Run
pip install -r requirements.txt
streamlit run app.py
📌 Output
Predicted price of the diamond
Market segment (cluster)
👤 Author

Dinesh
