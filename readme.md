# Customer Segmentation with BigQuery ML and Generative AI
<img width="287" height="293" alt="2 5" src="https://github.com/user-attachments/assets/81289e54-df71-4e9b-9aa3-c5a57de46b28" />


This project demonstrates an end-to-end data science workflow for customer segmentation.

It uses a public e-commerce dataset to perform RFM (Recency, Frequency, Monetary) analysis

Builds a K-Means clustering model using BigQuery ML, visualizes the resulting customer segments

Leverages the Gemini large language model to generate creative marketing personas and actionable strategies for each segment.

The entire workflow is executed within a Colab Enterprise Notebook in Google Cloud's BigQuery Studio, showcasing the seamless integration of data preparation, machine learning, and generative AI on a unified platform.

## 🚀 Technologies Used

# Google Cloud Platform (GCP)

# BigQuery (for data storage and ML model training)

# Vertex AI (for the Gemini Pro model)

# BigQuery Studio (for the notebook environment)


Python Libraries

bigframes.pandas: A library that provides a pandas-like API for working with large datasets in BigQuery.

bigframes.ml: A library for training machine learning models directly within BigQuery using a scikit-learn-like API.

google-cloud-bigquery: The official Python client for BigQuery.

google-cloud-aiplatform: The Vertex AI SDK for Python.

matplotlib: For data visualization.

## 📋 Project Workflow
The notebook follows these key steps:

# 1. Setup and Initialization
First, we import the necessary Python libraries and initialize the BigQuery and Vertex AI clients with the project details.

Python

from google.cloud import bigquery
from google.cloud import aiplatform
import bigframes.pandas as bpd
from vertexai.generative_models import GenerativeModel
from bigframes.ml.cluster import KMeans

# Define project variables
project_id = 'qwiklabs-gcp-03-c63e9e82159a'
dataset_name = "ecommerce"
location = "us-central1"

# Initialize clients
client = bigquery.Client(project=project_id)
aiplatform.init(project=project_id, location=location)

# 2. Data Preparation: RFM Analysis
A new table, ecommerce.customer_stats, is created using a BigQuery SQL query. This table calculates three key marketing metrics for each customer based on their order history from 2022:

Recency: days_since_last_order

Frequency: count_orders

Monetary: average_spend

SQL

CREATE OR REPLACE TABLE ecommerce.customer_stats AS
SELECT
  user_id,
  DATE_DIFF(CURRENT_DATE(), CAST(MAX(order_created_date) AS DATE), day) AS days_since_last_order, --RECENCY
  COUNT(order_id) AS count_orders, --FREQUENCY
  AVG(sale_price) AS average_spend --MONETARY
  FROM (
      SELECT
        user_id,
        order_id,
        sale_price,
        created_at AS order_created_date
        FROM `bigquery-public-data.thelook_ecommerce.order_items`
        WHERE
        created_at
            BETWEEN '2022-01-01' AND '2023-01-01'
  )
GROUP BY user_id;

# 3. K-Means Clustering Model
The prepared data is loaded into a BigQuery DataFrame, and a K-Means clustering model is trained to segment the customers into 5 distinct groups.

Python

# Load data into a BigQuery DataFrame
bqdf = bpd.read_gbq(f"{project_id}.{dataset_name}.{table_name}")

# Create and fit a K-Means model with 5 clusters
kmeans_model = KMeans(n_clusters=5)
kmeans_model.fit(bqdf)

# 4. Visualization
The model's predictions are used to create a scatter plot with matplotlib, visualizing the customer clusters based on their average spend and the days since their last order.

Python

# Use the model to predict clusters
predictions_df = kmeans_model.predict(bqdf)

# Generate scatterplot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    predictions_df["days_since_last_order"],
    predictions_df["average_spend"],
    c=predictions_df["CENTROID_ID"],
    cmap="viridis"
)
plt.xlabel("Days Since Last Order")
plt.ylabel("Average Spend")
plt.title("Attribute grouped by K-means cluster")
plt.colorbar(scatter, label="Cluster ID")
plt.show()
<img width="419" height="440" alt="2 6" src="https://github.com/user-attachments/assets/e747b079-9543-4f60-b92b-245a235be49e" />


# 5. Generative AI for Marketing Insights
The centroid data (the average RFM values for each cluster) is extracted from the trained model in BigQuery. This data is then formatted and passed to the Gemini Pro model with a specific prompt.

Prompt Snippet:

Python

prompt = f"""
You're a creative brand strategist, given the following clusters, come up with a \
creative brand persona, a catchy title, and next marketing action, \
explained step by step...

Clusters:
{cluster_info}

For each Cluster:
* Title:
* Persona:
* Next marketing step:
"""
The model generates detailed marketing personas, creative titles, and actionable next steps for each customer segment.

## 📊 Results
The output is a comprehensive, AI-generated marketing strategy tailored to each customer segment. An example of the output for one cluster is:

Cluster 1
Data: Average spend $318.16, count of orders 1.12, days since last order 1164.65

Title: The One-Time High Roller

Persona: Meet Eleanor, 55. Three years ago, she was likely making a significant, considered purchase. This wasn't an impulse buy; it was an investment... She values quality, is willing to pay for it, but isn't a frequent browser.

Next Marketing Step: The "Elevated Return" Campaign

This customer is not motivated by a simple "10% off" coupon. The goal is to remind them of the quality they invested in and entice them with an offer that matches their previous spending power.

Step 1: Craft a Premium, Personalized Email...

Step 2: Remind and Relate...

Step 3: Make a High-Value Offer...

⚙️ How to Run
Prerequisites: Ensure you have a Google Cloud project with the BigQuery and Vertex AI APIs enabled.

Environment: Open this .ipynb file in a Colab Enterprise Notebook within BigQuery Studio.

Configuration: Update the project_id and location variables in the second code cell to match your project details.


Execution: Run the notebook cells sequentially from top to bottom.
