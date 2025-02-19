import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from kneed import KneeLocator
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load the data
data_path = r"./Medical Inventory Optimaization Dataset.xlsx"
data = pd.read_excel(data_path)

# Data preprocessing
data['Dateofbill'] = pd.to_datetime(data['Dateofbill'])
data["Patient_ID"] = data["Patient_ID"].astype('str')
data["Final_Sales"] = data["Final_Sales"].astype('float32')
data["Final_Cost"] = data["Final_Cost"].astype('float32')
data = data.drop_duplicates()
data['Formulation'].fillna('Unknown', inplace=True)
data['SubCat'].fillna('Unknown', inplace=True)
data['SubCat1'].fillna('Unknown', inplace=True)
data['DrugName'] = data.groupby(['Specialisation', 'Dept'])['DrugName'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))
data.dropna(inplace=True)
data = data[data['Typeofsales'] != 'Return']
data = data.reset_index(drop=True)

# Map dates to monthly periods
data['Monthly_Period'] = data['Dateofbill'].apply(lambda date: date.to_period('M').to_timestamp())
monthly_data = data.groupby(['DrugName', 'Monthly_Period'])['Quantity'].sum().reset_index()
top_10_drugs = data.groupby('DrugName')['Quantity'].sum().nlargest(10).reset_index()
top_10_drug_names = top_10_drugs['DrugName'].tolist()
filtered_data = monthly_data[monthly_data['DrugName'].isin(top_10_drug_names)]
X = filtered_data[['Quantity']]
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters
wcss = []
for i in range(2, 11):
    kmeans = TimeSeriesKMeans(n_clusters=i, metric="dtw", max_iter=400, verbose=True, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
optimal_clusters = kl.knee

# Perform clustering with the optimal number of clusters
kmeans = TimeSeriesKMeans(n_clusters=optimal_clusters, metric="dtw")
clusters = kmeans.fit_predict(X_scaled)
filtered_data['Cluster'] = clusters
drug_cluster_mapping = filtered_data[['DrugName', 'Cluster']].drop_duplicates().set_index('DrugName')['Cluster'].to_dict()

# Forecast for each drug in the top 10 using LSTM
mape_values = []
individual_mape_values = {}
drug_data_with_forecast = {}

for drug in top_10_drug_names:
    drug_data = filtered_data[filtered_data['DrugName'] == drug]
    drug_data.set_index('Monthly_Period', inplace=True)

    if len(drug_data) > 2:
        scaler = MaxAbsScaler()
        drug_data_scaled = scaler.fit_transform(drug_data[['Quantity']])
        n_input = 2
        n_features = 1
        generator = TimeseriesGenerator(drug_data_scaled, drug_data_scaled, length=n_input, batch_size=1)
        split_idx = int(len(generator) * 0.8)
        train_generator = TimeseriesGenerator(drug_data_scaled[:split_idx], drug_data_scaled[:split_idx], length=n_input, batch_size=1)
        test_generator = TimeseriesGenerator(drug_data_scaled[split_idx:], drug_data_scaled[split_idx:], length=n_input, batch_size=1)
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(train_generator, epochs=100, verbose=0)
        predictions = model.predict(test_generator)
        predictions = scaler.inverse_transform(predictions)
        y_test = drug_data['Quantity'].values[-len(predictions):]

        if not np.any(np.isnan(predictions)) and not np.any(np.isnan(y_test)):
            mape = mean_absolute_percentage_error(y_test, predictions)
            mape_values.append(mape)
            individual_mape_values[drug] = mape
            drug_data_with_forecast[drug] = drug_data.reset_index()
            drug_data_with_forecast[drug]['Forecast'] = np.nan
            drug_data_with_forecast[drug].loc[drug_data_with_forecast[drug].index[-len(predictions):], 'Forecast'] = predictions

overall_mape = np.mean(mape_values) if mape_values else np.nan

# Streamlit application
st.title('Medical Inventory Optimization')

st.sidebar.header('Select a Drug')
selected_drug = st.sidebar.selectbox('Drug', top_10_drug_names)

st.write(f"*Overall Mean Absolute Percentage Error (MAPE):* {overall_mape:.2f}")

if selected_drug:
    st.header(f"Data for {selected_drug}")

    if selected_drug in drug_data_with_forecast:
        drug_forecast_data = drug_data_with_forecast[selected_drug]
        st.write(f"*Individual MAPE for {selected_drug}:* {individual_mape_values[selected_drug]:.2f}")

        st.write(f"*Key Data for {selected_drug}:*")
        latest_data = drug_forecast_data.tail(1).iloc[0]
        st.write(f"Monthly Period: {latest_data['Monthly_Period']}")
        st.write(f"Quantity: {latest_data['Quantity']}")
        st.write(f"Forecast: {latest_data['Forecast']}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(drug_forecast_data['Monthly_Period'], drug_forecast_data['Quantity'], label='Actual Quantity', marker='o')
        ax.plot(drug_forecast_data['Monthly_Period'], drug_forecast_data['Forecast'], label='Forecasted Quantity', marker='x')
        ax.set_xlabel('Date')
        ax.set_ylabel('Quantity Sold')
        ax.set_title(f'Actual vs Forecasted Quantity for {selected_drug}')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write(f"No forecast data available for {selected_drug}")

    st.write(f"*Return Quantity Data for {selected_drug}:*")
    selected_drug_data = data[data['DrugName'] == selected_drug]
    if 'ReturnQuantity' in data.columns:
        latest_return_data = selected_drug_data.tail(1).iloc[0]
        st.write(f"Monthly Period: {latest_return_data['Monthly_Period']}")
        st.write(f"Quantity: {latest_return_data['Quantity']}")
        st.write(f"Return Quantity: {latest_return_data['ReturnQuantity']}")
    else:
        latest_return_data = selected_drug_data.tail(1).iloc[0]
        st.write(f"Monthly Period: {latest_return_data['Monthly_Period']}")
        st.write(f"Quantity: {latest_return_data['Quantity']}")
