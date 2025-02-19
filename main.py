
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from kneed import KneeLocator
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_percentage_error, silhouette_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pickle

# Loading the data
data = pd.read_excel(r"./Medical Inventory Optimaization Dataset.xlsx")

# Convert 'Dateofbill' from string to datetime format
data['Dateofbill'] = pd.to_datetime(data['Dateofbill'])

# Type casting
data["Patient_ID"] = data["Patient_ID"].astype('str')
data["Final_Sales"] = data["Final_Sales"].astype('float32')
data["Final_Cost"] = data["Final_Cost"].astype('float32')

# Handling Duplicates
data = data.drop_duplicates()

#####       Handling missing values ##########

#  Fill the Missing values in 'Formulation', 'SubCat', and 'SubCat1' with "UNKNOWN"
data['Formulation'].fillna('Unknown', inplace=True)
data['SubCat'].fillna('Unknown', inplace=True)
data['SubCat1'].fillna('Unknown', inplace=True)

#  Fill missing 'DrugName' based on 'Specialisation' and 'Dept' groups
data['DrugName'] = data.groupby(['Specialisation', 'Dept'])['DrugName'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))


# Drop remaining missing values
data.dropna(inplace=True)
data = data.reset_index(drop=True)

# Remove rows with return type of transaction
data = data[data['Typeofsales'] != 'Return']

# Define a function to map dates to monthly periods
def get_monthly_period(date):
    return date.to_period('M').to_timestamp()

# Apply the function to create a monthly period column
data['Monthly_Period'] = data['Dateofbill'].apply(get_monthly_period)

# Aggregate sales data monthly
monthly_data = data.groupby(['DrugName', 'Monthly_Period'])['Quantity'].sum().reset_index()

# Find top 10 drugs based on total quantity sold
top_10_drugs = data.groupby('DrugName')['Quantity'].sum().nlargest(10).reset_index()

# Filter data to include only the top 10 drugs
top_10_drug_names = top_10_drugs['DrugName'].tolist()
filtered_data = monthly_data[monthly_data['DrugName'].isin(top_10_drug_names)]

# Prepare data for clustering
X = filtered_data[['Quantity']]

# Scale the data
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method and silhouette scores
wcss = []
silhouette_scores = []
for i in range(2, 11):  # Trying 2 to 10 clusters
    kmeans = TimeSeriesKMeans(n_clusters=i, metric="dtw", max_iter=400, verbose=True, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    cluster_labels = kmeans.predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels, metric="euclidean")
    silhouette_scores.append(silhouette_avg)

# Plot the Elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method For Optimal Clusters')
plt.grid(True)
plt.show()

# Plot the silhouette scores for optimal clusters
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores For Optimal Clusters')
plt.grid(True)
plt.show()

# Determine optimal clusters using the KneeLocator
kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
optimal_clusters = kl.knee

print(f'Optimal number of clusters: {optimal_clusters}')

# Perform clustering with the optimal number of clusters
kmeans = TimeSeriesKMeans(n_clusters=optimal_clusters, metric="dtw")
clusters = kmeans.fit_predict(X_scaled)
filtered_data['Cluster'] = clusters

# Plot clusters
for cluster in range(optimal_clusters):
    plt.figure(figsize=(10, 6))
    cluster_data = filtered_data[filtered_data['Cluster'] == cluster]
    for drug in cluster_data['DrugName'].unique():
        drug_data = cluster_data[cluster_data['DrugName'] == drug]
        plt.plot(drug_data['Monthly_Period'], drug_data['Quantity'], label=drug, marker='o')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.title(f'Cluster {cluster} Sales')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()

# Create a mapping of each drug to its cluster
drug_cluster_mapping = filtered_data[['DrugName', 'Cluster']].drop_duplicates().set_index('DrugName')['Cluster'].to_dict()


# Initialize lists to store individual MAPE values and drug data with forecast
mape_values = []
individual_mape_values = {}
drug_data_with_forecast = {}

# Forecast for each drug in the top 10 using LSTM
for drug in top_10_drug_names:
    drug_data = filtered_data[filtered_data['DrugName'] == drug]

    # Set 'Monthly_Period' as the index
    drug_data.set_index('Monthly_Period', inplace=True)
    
    # Check if there are enough data points
    if len(drug_data) > 2:  # Adjust this threshold if necessary
        # Normalize the data
        scaler = MaxAbsScaler()
        drug_data_scaled = scaler.fit_transform(drug_data[['Quantity']])
        
        # Define the time series generator
        n_input = 2  # Number of past months to use for predicting the future
        n_features = 1
        generator = TimeseriesGenerator(drug_data_scaled, drug_data_scaled, length=n_input, batch_size=1)
        
        # Split the data into training and testing sets
        split_idx = int(len(generator) * 0.8)
        train_generator = TimeseriesGenerator(drug_data_scaled[:split_idx], drug_data_scaled[:split_idx], length=n_input, batch_size=1)
        test_generator = TimeseriesGenerator(drug_data_scaled[split_idx:], drug_data_scaled[split_idx:], length=n_input, batch_size=1)
        
        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # Fit the model
        model.fit(train_generator, epochs=100, verbose=0)
        
        # Make predictions
        predictions = model.predict(test_generator)
        
        # Inverse transform the predictions
        predictions = scaler.inverse_transform(predictions)
        
        # Ensure predictions and actuals have no NaN values
        y_test = drug_data['Quantity'].values[-len(predictions):]
        if not np.any(np.isnan(predictions)) and not np.any(np.isnan(y_test)):
            # Calculate MAPE for this drug
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            # Append to the list of MAPE values
            mape_values.append(mape)
            individual_mape_values[drug] = mape
            
            # Store drug data with forecast
            drug_data_with_forecast[drug] = drug_data.reset_index()
            drug_data_with_forecast[drug]['Forecast'] = np.nan
            drug_data_with_forecast[drug].loc[drug_data_with_forecast[drug].index[-len(predictions):], 'Forecast'] = predictions
            
            # Plot the forecast
            plt.figure(figsize=(10, 6))
            plt.plot(drug_data.index[:-len(predictions)], drug_data['Quantity'].values[:-len(predictions)], label='Train')
            plt.plot(drug_data.index[-len(predictions):], y_test, label='Test')
            plt.plot(drug_data.index[-len(predictions):], predictions, label='Forecast')
            plt.xlabel('Date')
            plt.ylabel('Quantity Sold')
            plt.title(f'LSTM Forecast for {drug}')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print(f'Warning: NaN values found in predictions or actuals for drug: {drug}')

# Calculate overall MAPE value(AVG)
if mape_values:
    overall_mape = np.mean(mape_values)
else:
    overall_mape = np.nan
print(f'Overall Mean Absolute Percentage Error: {overall_mape}')

# Print individual MAPE values for each drug
print("Individual MAPE values for each drug:")
for drug, mape in individual_mape_values.items():
    print(f'{drug}: {mape}')

# Display quantity, return quantity, and forecast values for each drug
print("Quantity, Return Quantity, and Forecast Values:")
for drug, df in drug_data_with_forecast.items():
    print(f"Drug: {drug}")
    if 'ReturnQuantity' in df.columns:
        print(df[['Monthly_Period', 'Quantity', 'ReturnQuantity', 'Forecast']])
    else:
        print(df[['Monthly_Period', 'Quantity', 'Forecast']])

# Aggregate and print individual MAPE values for each drug by cluster
cluster_mape = {i: {} for i in range(optimal_clusters)}
for drug, mape in individual_mape_values.items():
    cluster = drug_cluster_mapping[drug]
    cluster_mape[cluster][drug] = mape

print("Individual MAPE values for each drug within each cluster:")
for cluster, drugs in cluster_mape.items():
    print(f'Cluster {cluster}:')
    for drug, mape in drugs.items():
        print(f'{drug}: {mape}')

# Filter and print drugs with MAPE values between 0.1 and 0.2
print("\nDrugs with MAPE values between 0.1 and 0.2:")
filtered_mape = {drug: mape for drug, mape in individual_mape_values.items() if 0.1 <= mape <= 0.2}
for drug, mape in filtered_mape.items():
    print(f'{drug}: {mape}')


# Save relevant objects using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)  # Save your trained LSTM model

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)  # Save your scaler object

with open('drug_cluster_mapping.pkl', 'wb') as f:
    pickle.dump(drug_cluster_mapping, f)  # Save your drug-cluster mapping dictionary

with open('filtered_mape.pkl', 'wb') as f:
    pickle.dump(filtered_mape, f)  # Save your filtered MAPE dictionary

