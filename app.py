import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("❤️ Heart Disease Prediction System")
st.markdown("""
This application predicts heart disease using 6 different machine learning models.
Upload your test data (CSV format) to get predictions and evaluate model performance.
""")

# Sidebar - Model Selection
st.sidebar.header("⚙️ Model Configuration")

model_options = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "K-Nearest Neighbors": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(model_options.keys())
)

# Load model and scaler
@st.cache_resource
def load_model(model_name):
    try:
        with open(model_options[model_name], 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

# File upload
st.sidebar.header("📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload test dataset with features and HeartDisease target column"
)

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.header("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        if 'HeartDisease' in df.columns:
            st.metric("Positive Cases", df['HeartDisease'].sum())
    
    # Show data preview
    with st.expander("🔍 View Data Sample"):
        st.dataframe(df.head(10))
    
    # Check if HeartDisease column exists
    if 'HeartDisease' not in df.columns:
        st.error("⚠️ Dataset must contain 'HeartDisease' column for evaluation!")
        st.stop()
    
    # Preprocessing
    try:
        # Separate features and target
        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]
        
        # One-hot encoding
        categorical_cols = X.select_dtypes(include=["object"]).columns
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Load scaler and transform
        scaler = load_scaler()
        if scaler is None:
            st.stop()
        
        # Get expected feature names from scaler
        expected_features = scaler.feature_names_in_
        
        # Align test data with training features
        # Add missing columns with 0s
        for col in expected_features:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        # Remove extra columns not in training
        X_encoded = X_encoded[expected_features]
        
        X_scaled = scaler.transform(X_encoded)
        X_scaled = pd.DataFrame(X_scaled, columns=expected_features)
        
        # Load selected model
        model = load_model(selected_model)
        if model is None:
            st.stop()
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        st.header(f"📈 Model Performance: {selected_model}")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
            st.metric("Precision", f"{precision_score(y, y_pred):.4f}")
        
        with metrics_col2:
            st.metric("ROC-AUC Score", f"{roc_auc_score(y, y_proba):.4f}")
            st.metric("Recall", f"{recall_score(y, y_pred):.4f}")
        
        with metrics_col3:
            st.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")
            st.metric("MCC Score", f"{matthews_corrcoef(y, y_pred):.4f}")
        
        # Confusion Matrix
        st.header("🎯 Confusion Matrix")
        
        cm = confusion_matrix(y, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Disease', 'Predicted Disease'],
            y=['Actual No Disease', 'Actual Disease'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            showscale=True
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {selected_model}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.header("📋 Classification Report")
        
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(
            report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
            use_container_width=True
        )
        
        # Prediction Distribution
        st.header("📊 Prediction Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            comparison_df = pd.DataFrame({
                'Actual': y.value_counts(),
                'Predicted': pd.Series(y_pred).value_counts()
            })
            
            fig = go.Figure(data=[
                go.Bar(name='Actual', x=['No Disease', 'Disease'], y=comparison_df['Actual'].values),
                go.Bar(name='Predicted', x=['No Disease', 'Disease'], y=comparison_df['Predicted'].values)
            ])
            
            fig.update_layout(
                title='Actual vs Predicted Distribution',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Probability distribution
            fig = px.histogram(
                x=y_proba,
                nbins=30,
                title='Prediction Probability Distribution',
                labels={'x': 'Probability of Heart Disease', 'y': 'Count'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Download predictions
        st.header("💾 Download Predictions")
        
        results_df = df.copy()
        results_df['Predicted_HeartDisease'] = y_pred
        results_df['Prediction_Probability'] = y_proba
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f'predictions_{selected_model.replace(" ", "_")}.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        st.info("Make sure your test data has the same features as the training data.")

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin.")
    
    # Display model comparison table
    st.header("📊 Model Performance Comparison")
    st.markdown("""
    Below is the performance comparison of all 6 implemented models on the Heart Disease dataset:
    """)
    
    # Create comparison table
    comparison_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 
                  'Random Forest', 'XGBoost'],
        'Accuracy': [0.8859, 0.7880, 0.8859, 0.9130, 0.8696, 0.8587],
        'ROC-AUC': [0.9297, 0.7813, 0.9360, 0.9451, 0.9314, 0.9219],
        'Precision': [0.8716, 0.7890, 0.8857, 0.9300, 0.8750, 0.8725],
        'Recall': [0.9314, 0.8431, 0.9118, 0.9118, 0.8922, 0.8725],
        'F1 Score': [0.9005, 0.8152, 0.8986, 0.9208, 0.8835, 0.8725],
        'MCC': [0.7694, 0.5691, 0.7686, 0.8246, 0.7356, 0.7140]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df.style.background_gradient(cmap='RdYlGn', subset=comparison_df.columns[1:]),
        use_container_width=True
    )
    
    st.success("🏆 Best Model: **Naive Bayes** (ROC-AUC: 0.9451)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>💡 Built with Streamlit | ML Assignment 2 | Heart Disease Prediction System</p>
</div>
""", unsafe_allow_html=True)
