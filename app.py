#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import logging
from PIL import Image
import io
import sys
sys.path.append(os.path.abspath("C:/Users/lenovo/.ipynb_checkpoints"))
# Import our product classifier
import product_classifier as pdcl

# Initialize the product classifier
product_classifier = pdcl.ProductClassifier()'''


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Set page configuration
st.set_page_config(
    page_title="Customer Spending Analysis",
    page_icon="💰",
    layout="wide",
)

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "clusters" not in st.session_state:
    st.session_state.clusters = None

# Page title
st.title("Customer Spending Analysis Dashboard")

# Function to load and prepare data
@st.cache_data
def load_data():
    """Load sample data or return empty DataFrame"""
    # Default sample data based on the provided overview
    data = {
        'CustomerID': [1, 2, 3, 4, 5],
        'Gender': ['Male', 'Male', 'Female', 'Female', 'Female'],
        'Age': [19, 21, 20, 23, 31],
        'Income': [15, 15, 16, 16, 17],
        'Spending': [39, 81, 6, 77, 40]
    }
    return pd.DataFrame(data)

# Function to train a simple spending prediction model
@st.cache_resource
def train_model(df):
    """Train a simple linear regression model for spending prediction"""
    # Prepare features
    X = df[['Age', 'Income']].copy()
    
    # Add gender as numeric
    X['Gender_Numeric'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    # Target variable
    y = df['Spending']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Function to perform customer segmentation
@st.cache_data
def perform_clustering(df, n_clusters=3):
    """Perform K-means clustering on customer data"""
    # Features for clustering
    features = ['Age', 'Income', 'Spending']
    X = df[features].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return df, kmeans, scaler

# Sidebar with options
with st.sidebar:
    st.header("Options")
    
    # Data source selection
    data_source = st.radio(
        "Select data source",
        ["Use sample data", "Upload your own CSV", "Enter data manually"]
    )
    
    # Add a cluster number slider in sidebar
    cluster_count = st.slider("Number of customer segments", 2, 5, 3)
    
    # Add information about the app
    st.markdown("---")
    st.info("""
    This dashboard analyzes customer spending patterns 
    based on demographic information.
    
    **Features:**
    - Spending prediction
    - Customer segmentation
    - Data visualization
    """)

# Load data based on selected source
if data_source == "Use sample data":
    df = load_data()
    st.session_state.df = df
    
elif data_source == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Check for required columns
            required_columns = ['CustomerID', 'Gender', 'Age', 'Income', 'Spending']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                st.session_state.df = df
                st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
elif data_source == "Enter data manually":
    st.sidebar.subheader("Create new customer record")
    
    # Form for manual data entry
    with st.sidebar.form("customer_form"):
        customer_id = st.number_input("Customer ID", min_value=1, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        income = st.number_input("Income (in thousands)", min_value=0, step=1)
        spending = st.number_input("Spending", min_value=0, step=1)
        
        submit_button = st.form_submit_button("Add Customer")
        
        if submit_button:
            new_customer = pd.DataFrame({
                'CustomerID': [customer_id],
                'Gender': [gender],
                'Age': [age],
                'Income': [income],
                'Spending': [spending]
            })
            
            if st.session_state.df is None:
                st.session_state.df = new_customer
            else:
                # Check if CustomerID already exists
                if customer_id in st.session_state.df['CustomerID'].values:
                    st.sidebar.warning("Customer ID already exists! Data updated.")
                    st.session_state.df = st.session_state.df[st.session_state.df['CustomerID'] != customer_id]
                    st.session_state.df = pd.concat([st.session_state.df, new_customer])
                else:
                    st.session_state.df = pd.concat([st.session_state.df, new_customer])
                    st.sidebar.success("Customer added!")

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Data Overview", "Spending Prediction", "Customer Segmentation"])

# Tab 1: Data Overview
with tab1:
    st.header("Customer Data Overview")
    
    if st.session_state.df is not None:
        # Display dataframe with styling
        st.dataframe(st.session_state.df, use_container_width=True)
        
        # Show basic statistics
        st.subheader("Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Customers", st.session_state.df.shape[0])
            st.metric("Average Age", f"{st.session_state.df['Age'].mean():.1f}")
            
        with col2:
            st.metric("Average Income", f"${st.session_state.df['Income'].mean():.1f}K")
            st.metric("Average Spending", f"${st.session_state.df['Spending'].mean():.1f}")
        
        # Create visualizations
        st.subheader("Visualizations")
        chart_type = st.radio("Select chart type", ["Distribution", "Correlation", "Gender Comparison"])
        
        if chart_type == "Distribution":
            # Create distribution plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Age distribution
            sns.histplot(st.session_state.df['Age'], kde=True, ax=axes[0])
            axes[0].set_title('Age Distribution')
            
            # Income distribution
            sns.histplot(st.session_state.df['Income'], kde=True, ax=axes[1])
            axes[1].set_title('Income Distribution')
            
            # Spending distribution
            sns.histplot(st.session_state.df['Spending'], kde=True, ax=axes[2])
            axes[2].set_title('Spending Distribution')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        elif chart_type == "Correlation":
            # Calculate correlation
            corr = st.session_state.df[['Age', 'Income', 'Spending']].corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            
        elif chart_type == "Gender Comparison":
            # Create comparison chart
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Spending by gender
            sns.barplot(x='Gender', y='Spending', data=st.session_state.df, ax=axes[0])
            axes[0].set_title('Average Spending by Gender')
            
            # Age vs. Spending by gender
            sns.scatterplot(x='Age', y='Spending', hue='Gender', data=st.session_state.df, ax=axes[1])
            axes[1].set_title('Age vs. Spending by Gender')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        # Allow downloading the data
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            "Download data as CSV",
            csv,
            "customer_data.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.info("Please select a data source to get started.")

# Tab 2: Spending Prediction
with tab2:
    st.header("Customer Spending Prediction")
    
    if st.session_state.df is not None and len(st.session_state.df) >= 3:
        # Train the model
        model = train_model(st.session_state.df)
        
        # Create form for prediction
        st.subheader("Predict Customer Spending")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_gender = st.radio("Gender", ["Male", "Female"])
            pred_age = st.slider("Age", 18, 100, 30)
            
        with col2:
            pred_income = st.slider("Income (in thousands)", 0, 200, 50)
            
        # Make prediction
        if st.button("Predict Spending"):
            # Create features
            X_pred = pd.DataFrame({
                'Age': [pred_age],
                'Income': [pred_income],
                'Gender_Numeric': [0 if pred_gender == 'Male' else 1]
            })
            
            # Predict
            prediction = model.predict(X_pred)[0]
            
            # Store prediction
            st.session_state.prediction_result = {
                'Gender': pred_gender,
                'Age': pred_age,
                'Income': pred_income,
                'Predicted_Spending': prediction
            }
            
            # Display prediction
            st.success(f"Predicted Spending: ${prediction:.2f}")
            
            # Show feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': ['Age', 'Income', 'Gender'],
                'Importance': abs(model.coef_)
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
            ax.set_title('Feature Importance for Spending Prediction')
            st.pyplot(fig)
            
            # Show similar customers
            st.subheader("Similar Customers")
            
            # Calculate distance based on Age and Income
            st.session_state.df['Distance'] = np.sqrt(
                (st.session_state.df['Age'] - pred_age)**2 + 
                (st.session_state.df['Income'] - pred_income)**2
            )
            
            # Get 3 most similar customers
            similar_customers = st.session_state.df.sort_values('Distance').head(3)
            st.dataframe(similar_customers[['CustomerID', 'Gender', 'Age', 'Income', 'Spending']])
    else:
        st.info("Need at least 3 customers to build a prediction model.")

# Tab 3: Customer Segmentation
with tab3:
    st.header("Customer Segmentation")
    
    if st.session_state.df is not None and len(st.session_state.df) >= cluster_count:
        # Perform clustering
        df_clustered, kmeans, scaler = perform_clustering(st.session_state.df, cluster_count)
        st.session_state.clusters = df_clustered
        
        # Display clusters
        st.subheader(f"Customer Segments ({cluster_count} clusters)")
        
        # Show cluster distribution
        cluster_counts = df_clustered['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Create pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(cluster_counts['Count'], labels=cluster_counts['Cluster'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
            
        with col2:
            # Show cluster characteristics
            cluster_stats = df_clustered.groupby('Cluster').agg({
                'Age': 'mean',
                'Income': 'mean',
                'Spending': 'mean'
            }).reset_index()
            
            # Round to 2 decimal places
            cluster_stats = cluster_stats.round(2)
            
            st.dataframe(cluster_stats, use_container_width=True)
            
        # Create scatter plot
        st.subheader("Cluster Visualization")
        plot_type = st.selectbox("Select visualization", ["Age vs. Spending", "Income vs. Spending", "Age vs. Income"])
        
        if plot_type == "Age vs. Spending":
            x_col, y_col = 'Age', 'Spending'
        elif plot_type == "Income vs. Spending":
            x_col, y_col = 'Income', 'Spending'
        else:
            x_col, y_col = 'Age', 'Income'
            
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(x=x_col, y=y_col, hue='Cluster', data=df_clustered, palette='viridis', s=100, ax=ax)
        ax.set_title(f'{x_col} vs. {y_col} by Cluster')
        
        # Add legend
        scatter.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        st.pyplot(fig)
        
        # Allow downloading the clustered data
        csv_clustered = df_clustered.to_csv(index=False)
        st.download_button(
            "Download clustered data as CSV",
            csv_clustered,
            "customer_segments.csv",
            "text/csv",
            key='download-clustered-csv'
        )
    else:
        st.info(f"Need at least {cluster_count} customers to perform clustering.")

# Footer
st.markdown("---")
st.caption("© 2025 Customer Spending Analysis Dashboard | Built with Streamlit")


# In[ ]:




