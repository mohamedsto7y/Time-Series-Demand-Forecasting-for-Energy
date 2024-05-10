import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import plotly.express as px

# Function to create time series features
def create_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Function to add lag features
def add_lags(df):
    # Add lag features
    df['lag1'] = df['feature_column'].shift(365)
    df['lag2'] = df['feature_column'].shift(730)
    df['lag3'] = df['feature_column'].shift(1095)
    return df

# Load model
reg = xgb.XGBRegressor()
reg.load_model('model.json')

# Streamlit app layout
def main():
    st.title('Energy Consumption Forecasting App')

    # Date input for future prediction
    date_input = st.date_input("Select a date for future prediction", datetime.now())

    # Create future date range
    future_date_range = pd.date_range(date_input, date_input + timedelta(hours=23), freq='1h')

    # Create DataFrame for future predictions
    future_df = pd.DataFrame(index=future_date_range)

    # Create features for future predictions
    future_df = create_features(future_df)
    future_df['feature_column'] = 0  # Placeholder for feature column used in lag creation

    # Add lag features for future predictions
    future_df = add_lags(future_df)

    # Make predictions
    features = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2', 'lag3']
    future_predictions = reg.predict(future_df[features])

    # Display predictions
    st.subheader('Future Energy Consumption Predictions')
    predictions_df = pd.DataFrame({'Datetime': future_date_range, 'Predicted Energy Consumption (MW)': future_predictions})
    st.write(predictions_df)

    # Time series graph for predictions using Plotly
    fig = px.line(predictions_df, x='Datetime', y='Predicted Energy Consumption (MW)', title='Predicted Energy Consumption')
    fig.update_xaxes(title='Datetime')
    fig.update_yaxes(title='Energy Consumption (MW)')
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
