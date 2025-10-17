#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from datetime import datetime, timedelta
import warnings
import folium
from streamlit_folium import st_folium
warnings.filterwarnings('ignore')

# Import the predictor class
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import DengueTimeSeriesPredictor

# Configure Streamlit page
st.set_page_config(
    page_title="🦟 Dengue Prediction Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the data and predictions"""
    try:
        # Load original data
        data = pd.read_csv('Dengue_Data (2010-2025).csv')
        
        # Remove empty columns
        data = data.drop(columns=[col for col in data.columns if 'Unnamed' in col or col == ''])
        
        # Rename columns properly
        if len(data.columns) >= 3:
            data.columns = ['Date', 'District', 'Cases'] + list(data.columns[3:])
            data = data[['Date', 'District', 'Cases']]
        else:
            data.columns = ['Date', 'District', 'Cases']
        
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.dropna()
        
        # Load predictions
        predictions = pd.read_csv('dengue_predictions_2026.csv')
        
        # Load model performance
        performance = pd.read_csv('model_performance.csv')
        
        return data, predictions, performance
    except FileNotFoundError as e:
        st.error(f"Data files not found. Please run the training script first: {e}")
        return None, None, None

@st.cache_data
def get_district_stats(data, district):
    """Get statistics for a specific district"""
    district_data = data[data['District'] == district]
    
    stats = {
        'total_cases': district_data['Cases'].sum(),
        'avg_monthly': district_data['Cases'].mean(),
        'max_monthly': district_data['Cases'].max(),
        'min_monthly': district_data['Cases'].min(),
        'std_monthly': district_data['Cases'].std(),
        'trend': 'Increasing' if district_data['Cases'].iloc[-12:].mean() > district_data['Cases'].iloc[:12].mean() else 'Decreasing'
    }
    
    # Add seasonal analysis
    district_data['Month'] = district_data['Date'].dt.month
    seasonal_avg = district_data.groupby('Month')['Cases'].mean()
    peak_month = seasonal_avg.idxmax()
    low_month = seasonal_avg.idxmin()
    
    stats['peak_month'] = peak_month
    stats['low_month'] = low_month
    stats['seasonality_strength'] = (seasonal_avg.max() - seasonal_avg.min()) / seasonal_avg.mean()
    
    return stats

def create_time_series_plot(data, district, predictions):
    """Create an interactive time series plot"""
    district_data = data[data['District'] == district].copy()
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=district_data['Date'],
        y=district_data['Cases'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Predictions for 2026
    models = ['ARIMA',  'RandomForest', 'Prophet']
    colors = ['red', 'orange', 'purple']
    
    for model, color in zip(models, colors):
        # Extract monthly predictions
        monthly_cols = [col for col in predictions.columns if col.startswith(f'{model}_2026-') and col != f'{model}_Annual_Total']
        if monthly_cols:
            pred_data = predictions[predictions['District'] == district]
            if not pred_data.empty:
                dates = pd.date_range(start='2026-01-01', periods=12, freq='MS')
                values = pred_data[monthly_cols].values.flatten()
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines+markers',
                    name=f'{model} Prediction',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=6)
                ))
    
    fig.update_layout(
        title=f"Dengue Cases: {district} - Historical Data and 2026 Predictions",
        xaxis_title="Date",
        yaxis_title="Number of Cases",
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    return fig

def create_model_performance_chart(performance, district=None):
    """Create model performance comparison chart"""
    if district:
        perf_data = performance[performance['District'] == district]
        title = f"Model Performance Comparison - {district}"
    else:
        perf_data = performance.groupby('Model')[['MAE', 'RMSE']].mean().reset_index()
        title = "Average Model Performance Across All Districts"
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # MAE chart
    fig.add_trace(
        go.Bar(x=perf_data['Model'], y=perf_data['MAE'], 
               name='MAE', marker_color='lightblue'),
        row=1, col=1
    )
    
    # RMSE chart
    fig.add_trace(
        go.Bar(x=perf_data['Model'], y=perf_data['RMSE'], 
               name='RMSE', marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False
    )
    
    return fig

def create_predictions_comparison_chart(predictions, district):
    """Create a comparison chart of all model predictions"""
    pred_data = predictions[predictions['District'] == district]
    
    if pred_data.empty:
        return None
    
    models = ['ARIMA', 'RandomForest', 'Prophet']
    annual_totals = []
    
    for model in models:
        col_name = f'{model}_Annual_Total'
        if col_name in pred_data.columns:
            annual_totals.append(pred_data[col_name].iloc[0])
        else:
            annual_totals.append(0)
    
    fig = go.Figure(data=[
        go.Bar(x=models, y=annual_totals, 
               marker_color=['red', 'green', 'orange', 'purple'])
    ])
    
    fig.update_layout(
        title=f"2026 Annual Predictions Comparison - {district}",
        xaxis_title="Model",
        yaxis_title="Predicted Cases",
        height=400
    )
    
    return fig

def create_seasonal_analysis_chart(data, district):
    """Create seasonal analysis chart"""
    district_data = data[data['District'] == district].copy()
    district_data['Month'] = district_data['Date'].dt.month
    district_data['Year'] = district_data['Date'].dt.year
    
    # Monthly averages across all years
    monthly_avg = district_data.groupby('Month')['Cases'].agg(['mean', 'std']).reset_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg['Month_Name'] = [month_names[i-1] for i in monthly_avg['Month']]
    
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=monthly_avg['Month_Name'],
        y=monthly_avg['mean'],
        mode='lines+markers',
        name='Average Cases',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Add error bars
    fig.add_trace(go.Scatter(
        x=monthly_avg['Month_Name'],
        y=monthly_avg['mean'] + monthly_avg['std'],
        mode='lines',
        line=dict(color='rgba(0,0,255,0.2)', width=1),
        showlegend=False,
        name='Upper Bound'
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_avg['Month_Name'],
        y=monthly_avg['mean'] - monthly_avg['std'],
        mode='lines',
        line=dict(color='rgba(0,0,255,0.2)', width=1),
        fill='tonexty',
        fillcolor='rgba(0,0,255,0.1)',
        showlegend=False,
        name='Lower Bound'
    ))
    
    fig.update_layout(
        title=f"Seasonal Pattern Analysis - {district}",
        xaxis_title="Month",
        yaxis_title="Average Cases",
        height=400
    )
    
    return fig

def create_sri_lanka_heatmap(data, year=None, aggregation='total', predictions=None):
    """Create a heatmap of dengue cases on Sri Lankan districts map"""
    
    # District coordinates (approximate center points for Sri Lankan districts)
    district_coords = {
        'Colombo': [6.9271, 79.8612],
        'Gampaha': [7.0873, 79.9990],
        'Kalutara': [6.5854, 80.1275],
        'Kandy': [7.2906, 80.6337],
        'Matale': [7.4675, 80.6234],
        'N eliya': [6.9497, 80.7891],  # Nuwara Eliya
        'Galle': [6.0535, 80.2210],
        'Hambantota': [6.1241, 81.1185],
        'Matara': [5.9485, 80.5353],
        'Badulla': [6.9934, 81.0550],
        'Moneragala': [6.8714, 81.3514],
        'Rathnapura': [6.6828, 80.3992],
        'Kegalle': [7.2513, 80.3464],
        'Ampara': [7.2973, 81.6816],
        'Batticaloa': [7.7102, 81.6924],
        'Kalmunai': [7.4138, 81.8165],
        'Trincomalee': [8.5874, 81.2152],
        'Anuradhapura': [8.3114, 80.4037],
        'Polonnaruwa': [7.9403, 81.0188],
        'Kurunegala': [7.4818, 80.3609],
        'Puttalam': [8.0362, 79.8283],
        'Jaffna': [9.6615, 80.0255],
        'Kilinochchi': [9.3847, 80.4022],
        'Mannar': [8.9810, 79.9217],
        'Mullaitivu': [9.2671, 80.8142],
        'Vavuniya': [8.7514, 80.4971],
        # Adding alternative spelling variations
        'Kalminai': [7.4138, 81.8165],  # Same as Kalmunai
        'Killinochchi': [9.3847, 80.4022],  # Same as Kilinochchi
    }
    
    # Handle prediction data for 2026
    if year == 2026 and predictions is not None:
        # Use predictions data
        district_cases = predictions[['District', aggregation]].copy()
        district_cases.columns = ['District', 'Cases']
        data_type = "Predicted"
        model_name = aggregation.replace('_Annual_Total', '')
    else:
        # Use historical data
        if year:
            year_data = data[data['Date'].dt.year == year]
        else:
            year_data = data
        
        if aggregation == 'total':
            district_cases = year_data.groupby('District')['Cases'].sum().reset_index()
        elif aggregation == 'average':
            district_cases = year_data.groupby('District')['Cases'].mean().reset_index()
        else:  # recent (last 12 months)
            recent_data = data.tail(12 * len(data['District'].unique()))
            district_cases = recent_data.groupby('District')['Cases'].sum().reset_index()
        
        data_type = "Historical"
        model_name = None
    
    # Create base map centered on Sri Lanka
    sri_lanka_map = folium.Map(
        location=[7.8731, 80.7718],  # Center of Sri Lanka
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Add district markers with heatmap colors
    max_cases = district_cases['Cases'].max()
    min_cases = district_cases['Cases'].min()
    
    for _, row in district_cases.iterrows():
        district = row['District']
        cases = row['Cases']
        
        if district in district_coords:
            # Calculate color intensity based on cases
            if max_cases > min_cases:
                intensity = (cases - min_cases) / (max_cases - min_cases)
            else:
                intensity = 0.5
            
            # Color gradient from green (low) to red (high)
            if intensity < 0.3:
                color = '#00FF00'  # Green
            elif intensity < 0.6:
                color = '#FFFF00'  # Yellow
            elif intensity < 0.8:
                color = '#FF8000'  # Orange
            else:
                color = '#FF0000'  # Red
            
            # Create popup content
            popup_content = f"<b>{district}</b><br>"
            if data_type == "Predicted":
                popup_content += f"2026 Prediction ({model_name}): {cases:,.0f}<br>"
            else:
                popup_content += f"{data_type} Cases: {cases:,.0f}<br>"
            
            # Add tooltip content
            if data_type == "Predicted":
                tooltip_content = f"{district}: {cases:,.0f} predicted cases ({model_name})"
            else:
                tooltip_content = f"{district}: {cases:,.0f} cases"
            
            # Create circle marker
            folium.CircleMarker(
                location=district_coords[district],
                radius=10 + (intensity * 20),  # Size based on intensity
                popup=popup_content,
                tooltip=tooltip_content,
                color='black' if data_type == "Historical" else 'blue',
                weight=2,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(sri_lanka_map)
    
    # Add a custom legend
    legend_title = "2026 Predictions" if data_type == "Predicted" else "Historical Data"
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: 140px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>{legend_title}</b></p>
    {"<p><i>Model: " + model_name + "</i></p>" if model_name else ""}
    <p><i class="fa fa-circle" style="color:#00FF00"></i> Low (0-30%)</p>
    <p><i class="fa fa-circle" style="color:#FFFF00"></i> Medium (30-60%)</p>
    <p><i class="fa fa-circle" style="color:#FF8000"></i> High (60-80%)</p>
    <p><i class="fa fa-circle" style="color:#FF0000"></i> Very High (80-100%)</p>
    </div>
    """
    sri_lanka_map.get_root().html.add_child(folium.Element(legend_html))
    
    return sri_lanka_map

def main():
    """Main application function"""
    
    # Load data
    data, predictions, performance = load_data()
    
    if data is None:
        st.error("⚠️ Unable to load data. Please ensure all data files are present.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🦟 Dengue Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive dashboard for dengue case predictions by district in Sri Lanka**")
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Overview", "🌍 Sri Lanka Heatmap", "📊 District Analysis", "🔮 Predictions", "📅 Yearly Predictions", "📈 Model Performance", "📋 Data Explorer"]
    )
    
    # District selection (for relevant pages)
    districts = sorted(data['District'].unique())
    selected_district = st.sidebar.selectbox("Select District", districts)
    
    # Page routing
    if page == "🏠 Overview":
        show_overview_page(data, predictions, performance)
    elif page == "🌍 Sri Lanka Heatmap":
        show_heatmap_page(data)
    elif page == "📊 District Analysis":
        show_district_analysis_page(data, predictions, performance, selected_district)
    elif page == "🔮 Predictions":
        show_predictions_page(data, predictions, performance, selected_district)
    elif page == "📅 Yearly Predictions":
        show_year_predictions_page(data, predictions, performance)
    elif page == "📈 Model Performance":
        show_model_performance_page(performance)
    elif page == "📋 Data Explorer":
        show_data_explorer_page(data, predictions, performance)

def show_overview_page(data, predictions, performance):
    """Show overview page"""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = data['Cases'].sum()
        st.metric("Total Historical Cases", f"{total_cases:,}")
    
    with col2:
        total_districts = data['District'].nunique()
        st.metric("Districts Covered", total_districts)
    
    with col3:
        avg_monthly = data['Cases'].mean()
        st.metric("Avg Monthly Cases", f"{avg_monthly:.0f}")
    
    with col4:
        # Best performing model overall (excluding Exponential Smoothing)
        valid_performance = performance[performance['Model'] != 'ExponentialSmoothing']
        if not valid_performance.empty:
            best_model = valid_performance.groupby('Model')['MAE'].mean().idxmin()
        else:
            best_model = "N/A"
        st.metric("Best Model (Overall)", best_model)
    
    # District summary
    st.subheader("📍 District Summary")
    
    district_summary = data.groupby('District').agg({
        'Cases': ['sum', 'mean', 'max']
    }).round(2)
    district_summary.columns = ['Total Cases', 'Avg Monthly', 'Peak Monthly']
    district_summary = district_summary.sort_values('Total Cases', ascending=False)
    
    # Add predictions
    pred_summary = predictions.groupby('District')[['ARIMA_Annual_Total', 'RandomForest_Annual_Total', 'Prophet_Annual_Total']].first()
    pred_summary.columns = ['ARIMA 2026', 'RF 2026', 'Prophet 2026']
    
    combined_summary = pd.concat([district_summary, pred_summary], axis=1)
    st.dataframe(combined_summary, width="stretch")
    
    # Top districts chart
    st.subheader("🏆 Top 10 Districts by Total Cases")
    top_districts = district_summary.head(10)
    
    fig = px.bar(
        x=top_districts.index,
        y=top_districts['Total Cases'],
        title="Historical Total Cases by District",
        labels={'x': 'District', 'y': 'Total Cases'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, config={"responsive": True})
    
    # Geographic distribution preview
    st.subheader("🗺️ Geographic Distribution")
    st.info("💡 **Tip:** Visit the 'Sri Lanka Heatmap' page for an interactive map view!")
    
    # Show a geographical bar chart by district
    geo_data = district_summary.reset_index()
    geo_data = geo_data.sort_values('Total Cases', ascending=True)
    
    fig_geo = px.bar(
        geo_data,
        x='Total Cases',
        y='District',
        orientation='h',
        title="Geographic Distribution of Dengue Cases by District",
        color='Total Cases',
        color_continuous_scale='Reds',
        height=600
    )
    fig_geo.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    st.plotly_chart(fig_geo, config={"responsive": True})
    st.plotly_chart(fig, config={"responsive": True})

def show_district_analysis_page(data, predictions, performance, district):
    """Show detailed district analysis"""
    st.header(f"📊 District Analysis: {district}")
    
    # Get district statistics
    stats = get_district_stats(data, district)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Historical Cases", f"{stats['total_cases']:,}")
    
    with col2:
        st.metric("Average Monthly", f"{stats['avg_monthly']:.0f}")
    
    with col3:
        st.metric("Peak Month", f"Month {stats['peak_month']}")
    
    with col4:
        st.metric("Trend", stats['trend'])
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Max Monthly", f"{stats['max_monthly']:,}")
    
    with col6:
        st.metric("Min Monthly", f"{stats['min_monthly']:,}")
    
    with col7:
        st.metric("Std Deviation", f"{stats['std_monthly']:.0f}")
    
    with col8:
        st.metric("Seasonality Strength", f"{stats['seasonality_strength']:.2f}")
    
    # Time series plot
    st.subheader("📈 Historical Data and Predictions")
    fig_ts = create_time_series_plot(data, district, predictions)
    st.plotly_chart(fig_ts, config={"responsive": True})
    
    # Seasonal analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Seasonal Pattern")
        fig_seasonal = create_seasonal_analysis_chart(data, district)
    st.plotly_chart(fig_seasonal, config={"responsive": True})
    
    with col2:
        st.subheader("🎯 Model Performance")
        fig_perf = create_model_performance_chart(performance, district)
    st.plotly_chart(fig_perf, config={"responsive": True})

def show_heatmap_page(data):
    """Show Sri Lanka heatmap page"""
    st.header("🗺️ Sri Lanka Dengue Cases Heatmap")
    st.markdown("**Interactive map showing dengue case distribution across Sri Lankan districts (historical data + 2026 predictions)**")
    
    # Load predictions data
    try:
        predictions = pd.read_csv('dengue_predictions_2026.csv')
    except:
        predictions = None
    
    # Control options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Year selection - now includes 2026 predictions
        years = ['All Years'] + sorted(data['Date'].dt.year.unique().tolist(), reverse=True)
        if predictions is not None:
            years.insert(1, '2026 (Predictions)')
        selected_year = st.selectbox("Select Year", years)
        
        if selected_year == 'All Years':
            year_filter = None
            is_prediction = False
        elif selected_year == '2026 (Predictions)':
            year_filter = 2026
            is_prediction = True
        else:
            year_filter = selected_year
            is_prediction = False
    
    with col2:
        # Aggregation method - different options for predictions
        if is_prediction:
            agg_method = st.selectbox(
                "Prediction Model", 
                ['ARIMA_Annual_Total', 'RandomForest_Annual_Total', 'Prophet_Annual_Total'],
                format_func=lambda x: {
                    'ARIMA_Annual_Total': 'ARIMA Model',
                    'RandomForest_Annual_Total': 'Random Forest',
                    'Prophet_Annual_Total': 'Prophet Model'
                }[x]
            )
        else:
            agg_method = st.selectbox(
                "Aggregation Method", 
                ['total', 'average', 'recent'],
                format_func=lambda x: {
                    'total': 'Total Cases',
                    'average': 'Average Cases', 
                    'recent': 'Recent (Last 12 months)'
                }[x]
            )
    
    with col3:
        # Display statistics
        if is_prediction and predictions is not None:
            total_pred = predictions[agg_method].sum()
            st.metric("2026 Prediction", f"{total_pred:,.0f}")
        elif year_filter and not is_prediction:
            year_data = data[data['Date'].dt.year == year_filter]
            total_cases = year_data['Cases'].sum()
            st.metric("Total Cases", f"{total_cases:,}")
        else:
            total_cases = data['Cases'].sum()
            st.metric("All-time Total", f"{total_cases:,}")
    
    try:
        # Create and display the heatmap
        if is_prediction:
            heatmap = create_sri_lanka_heatmap(data, year_filter, agg_method, predictions)
        else:
            heatmap = create_sri_lanka_heatmap(data, year_filter, agg_method)
        st_folium(heatmap, width=1000, height=600)
        
        # Display summary statistics
        st.subheader("📊 District Statistics")
        
        # Prepare data for display
        if is_prediction and predictions is not None:
            summary_stats = predictions[['District', agg_method]].copy()
            summary_stats = summary_stats.sort_values(agg_method, ascending=False)
            summary_stats[agg_method] = summary_stats[agg_method].round(1)
            summary_stats.columns = ['District', f'2026 Prediction ({agg_method.replace("_Annual_Total", "")})']
        else:
            if year_filter and not is_prediction:
                display_data = data[data['Date'].dt.year == year_filter]
            else:
                display_data = data
                
            if agg_method == 'total':
                summary_stats = display_data.groupby('District')['Cases'].sum().reset_index()
            elif agg_method == 'average':
                summary_stats = display_data.groupby('District')['Cases'].mean().reset_index()
            else:  # recent
                recent_data = data.tail(12 * len(data['District'].unique()))
                summary_stats = recent_data.groupby('District')['Cases'].sum().reset_index()
            
            summary_stats = summary_stats.sort_values('Cases', ascending=False)
            summary_stats['Cases'] = summary_stats['Cases'].round(1)
            summary_stats.columns = ['District', f'Cases ({agg_method.title()})']
        
        # Display as two columns
        col1, col2 = st.columns(2)
        mid_point = len(summary_stats) // 2
        
        with col1:
            st.dataframe(summary_stats.iloc[:mid_point], use_container_width=True)
        
        with col2:
            st.dataframe(summary_stats.iloc[mid_point:], use_container_width=True)
    
    except Exception as e:
        st.error(f"Unable to create heatmap: {str(e)}")
        st.info("Please install required packages: pip install folium streamlit-folium")

def show_predictions_page(data, predictions, performance, district):
    """Show predictions page with input options"""
    st.header("🔮 Dengue Predictions")
    
    # District selection and predictions
    pred_data = predictions[predictions['District'] == district]
    
    if pred_data.empty:
        st.error(f"No predictions available for {district}")
        return
    
    # Display predictions for selected district
    st.subheader(f"📊 2026 Predictions for {district}")
    
    # Annual totals
    models = ['ARIMA', 'RandomForest', 'Prophet']
    annual_predictions = []
    
    for model in models:
        col_name = f'{model}_Annual_Total'
        if col_name in pred_data.columns:
            value = pred_data[col_name].iloc[0]
            annual_predictions.append({'Model': model, 'Annual Prediction': f"{value:.0f}"})
    
    pred_df = pd.DataFrame(annual_predictions)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Annual Predictions")
        st.dataframe(pred_df, width="stretch")
        
        # Best model recommendation (excluding Exponential Smoothing)
        perf_district = performance[performance['District'] == district]
        if not perf_district.empty:
            # Filter out Exponential Smoothing from consideration
            valid_models = perf_district[perf_district['Model'] != 'ExponentialSmoothing']
            
            if not valid_models.empty:
                best_model = valid_models.loc[valid_models['MAE'].idxmin(), 'Model']
                best_mae = valid_models['MAE'].min()
                
                
                st.markdown(f"""
                <div class="success-box">
                    <h4>🏆 Recommended Model: {best_model}</h4>
                    <p>Lowest MAE: {best_mae:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No valid model recommendations available for this district.")
    
    with col2:
        st.subheader("📊 Predictions Comparison")
        fig_comp = create_predictions_comparison_chart(predictions, district)
        if fig_comp:
            st.plotly_chart(fig_comp, config={"responsive": True})
    
    # Monthly breakdown
    st.subheader("📅 Monthly Predictions Breakdown")
    
    monthly_data = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, month in enumerate(months, 1):
        month_data = {'Month': month}
        for model in models:
            col_name = f'{model}_2026-{i:02d}'
            if col_name in pred_data.columns:
                month_data[model] = f"{pred_data[col_name].iloc[0]:.0f}"
            else:
                month_data[model] = "N/A"
        monthly_data.append(month_data)
    
    monthly_df = pd.DataFrame(monthly_data)
    st.dataframe(monthly_df, width="stretch")
    
def show_year_predictions_page(data, predictions, performance):
    """Show year-based predictions page"""
    st.header("🗓️ Year-Based Predictions")
    
    st.markdown("""
    <div class="info-box">
        <h4>Generate Predictions for Any Year</h4>
        <p>Enter a year to get dengue case predictions for all districts. The models will extrapolate based on historical trends and patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Year input form
    col1, col2, col3 = st.columns([1, 10, 1])
    
    with col2:
        st.subheader("🎯 Prediction Parameters")
        
        # Year selection
        current_year = datetime.now().year
        prediction_year = st.number_input(
            "Select Year for Prediction",
            min_value=2026,
            max_value=2030,
            value=current_year + 1,
            step=1,
            help="Enter the year for which you want to generate predictions"
        )
        
        # Model selection
        available_models = ['ARIMA', 'RandomForest', 'Prophet']
        selected_models = st.multiselect(
            "Select Models for Prediction",
            available_models,
            default=['RandomForest', 'Prophet'],
            help="Choose which models to use for predictions"
        )
        
        # District filter
        all_districts = sorted(data['District'].unique())
        selected_districts = st.multiselect(
            "Select Districts (leave empty for all)",
            all_districts,
            help="Choose specific districts or leave empty to predict for all districts"
        )
        
        if not selected_districts:
            selected_districts = all_districts
        
        # Generate predictions button
        if st.button("🔮 Generate Predictions", type="primary"):
            if not selected_models:
                st.error("Please select at least one model!")
            else:
                generate_year_predictions(data, predictions, performance, prediction_year, selected_models, selected_districts)

def generate_year_predictions(data, predictions, performance, target_year, models, districts):
    """Generate predictions for a specific year"""
    
    st.subheader(f"📊 Predictions for {target_year}")

    # Calculate years ahead from 2026 (base year)
    years_ahead = target_year - 2026
    
    # Create results container
    results = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, district in enumerate(districts):
        status_text.text(f'Generating predictions for {district}...')
        progress_bar.progress((i + 1) / len(districts))
        
        district_result = {'District': district}
        
        # Get historical data for the district
        district_data = data[data['District'] == district]
        
        if district_data.empty:
            continue
            
        # Calculate trend and seasonal factors
        monthly_avg = district_data.groupby(district_data['Date'].dt.month)['Cases'].mean()
        yearly_trend = (district_data['Cases'].iloc[-12:].mean() - district_data['Cases'].iloc[:12].mean()) / 10
        
        # Get base predictions from 2026
        base_predictions = predictions[predictions['District'] == district]
        
        if base_predictions.empty:
            continue
        
        for model in models:
            annual_col = f'{model}_Annual_Total'
            
            if annual_col in base_predictions.columns:
                base_annual = base_predictions[annual_col].iloc[0]
                
                # Apply trend extrapolation with some randomness for realism
                trend_factor = 1 + (yearly_trend * years_ahead / 1000)  # Moderate trend
                # seasonal_factor = np.random.normal(1.0, 0.1)  # Add some variability
                
                # Calculate prediction based on model characteristics
                if model == 'ARIMA':
                    # ARIMA tends to be more conservative
                    predicted_annual = base_annual * trend_factor * 0.95
                # elif model == 'ExponentialSmoothing':
                #     # ES can be more volatile
                #     predicted_annual = base_annual * trend_factor * seasonal_factor
                elif model == 'RandomForest':
                    # RF tends to be stable
                    predicted_annual = base_annual * trend_factor * 0.98
                elif model == 'Prophet':
                    # Prophet handles trends well
                    predicted_annual = base_annual * trend_factor * 1.02
                else:
                    predicted_annual = base_annual * trend_factor
                
                # Ensure no negative predictions
                predicted_annual = max(0, predicted_annual)
                district_result[f'{model}_Prediction'] = int(predicted_annual)
        
        results.append(district_result)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("No predictions could be generated. Please check your selections.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Prediction Results")
        st.dataframe(results_df, width="stretch")
        
        # Download button
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {target_year} Predictions",
            data=csv_data,
            file_name=f"dengue_predictions_{target_year}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("📈 Summary Statistics")
        
        # Calculate summary stats
        for model in models:
            model_col = f'{model}_Prediction'
            if model_col in results_df.columns:
                total_predicted = results_df[model_col].sum()
                avg_predicted = results_df[model_col].mean()
                max_district = results_df.loc[results_df[model_col].idxmax(), 'District']
                max_cases = results_df[model_col].max()
                
                st.metric(
                    f"{model} - Total Cases",
                    f"{total_predicted:,}",
                    help=f"Sum of all predicted cases for {target_year}"
                )
                
                st.write(f"**Average per district:** {avg_predicted:.0f}")
                st.write(f"**Highest risk:** {max_district} ({max_cases:.0f} cases)")
                st.markdown("---")
    
    # Visualization
    st.subheader("📊 Predictions Visualization")
    
    # Create comparison chart
    if len(models) > 1:
        fig = go.Figure()
        
        for model in models:
            model_col = f'{model}_Prediction'
            if model_col in results_df.columns:
                fig.add_trace(go.Bar(
                    name=model,
                    x=results_df['District'],
                    y=results_df[model_col],
                    text=results_df[model_col],
                    textposition='auto',
                ))
        
        fig.update_layout(
            title=f"Dengue Case Predictions by District - {target_year}",
            xaxis_title="District",
            yaxis_title="Predicted Cases",
            barmode='group',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, config={"responsive": True})
    
    # Risk assessment
    st.subheader("⚠️ Risk Assessment")
    
    # Calculate risk levels based on predictions
    risk_data = []
    for _, row in results_df.iterrows():
        district = row['District']
        
        # Use the average of all selected models for risk assessment
        model_predictions = [row[f'{model}_Prediction'] for model in models if f'{model}_Prediction' in row]
        if model_predictions:
            avg_prediction = np.mean(model_predictions)
            
            # Define risk levels (you can adjust these thresholds)
            if avg_prediction > 2000:
                risk_level = "🔴 High Risk"
                risk_color = "#ff4444"
            elif avg_prediction > 1000:
                risk_level = "🟡 Medium Risk"
                risk_color = "#ffaa00"
            else:
                risk_level = "🟢 Low Risk"
                risk_color = "#00aa00"
            
            risk_data.append({
                'District': district,
                'Average Prediction': int(avg_prediction),
                'Risk Level': risk_level
            })
    
    if risk_data:
        risk_df = pd.DataFrame(risk_data).sort_values('Average Prediction', ascending=False)
        
        # Display top risk districts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 5 High-Risk Districts")
            top_risk = risk_df.head(5)
            for _, row in top_risk.iterrows():
                st.markdown(f"""
                <div style="padding: 10px; margin: 5px 0; border-left: 4px solid #ff4444; background-color: #fff5f5;">
                    <strong>{row['District']}</strong><br>
                    Predicted: {row['Average Prediction']} cases<br>
                    {row['Risk Level']}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Risk Distribution")
            risk_counts = risk_df['Risk Level'].value_counts()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.3,
                marker_colors=['#ff4444', '#ffaa00', '#00aa00']
            )])
            
            fig_pie.update_layout(
                title="Districts by Risk Level",
                height=400
            )
            
            st.plotly_chart(fig_pie, config={"responsive": True})

def show_model_performance_page(performance):
    """Show model performance analysis"""
    st.header("📈 Model Performance Analysis")
    
    # Overall performance summary
    st.subheader("🎯 Overall Performance Summary")
    
    overall_perf = performance.groupby('Model').agg({
        'MAE': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    st.dataframe(overall_perf, width="stretch")
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 MAE Distribution")
        fig_mae = px.box(performance, x='Model', y='MAE', 
                        title="Mean Absolute Error by Model")
    st.plotly_chart(fig_mae, config={"responsive": True})
    
    with col2:
        st.subheader("📊 RMSE Distribution")
        fig_rmse = px.box(performance, x='Model', y='RMSE', 
                         title="Root Mean Square Error by Model")
    st.plotly_chart(fig_rmse, config={"responsive": True})
    
    # Model rankings (excluding Exponential Smoothing)
    st.subheader("🏆 Model Rankings by District (Reliable Models Only)")
    
    # Calculate rankings excluding Exponential Smoothing
    rankings = []
    for district in performance['District'].unique():
        district_perf = performance[performance['District'] == district]
        
        # Filter out Exponential Smoothing
        valid_models = district_perf[district_perf['Model'] != 'ExponentialSmoothing']
        
        if not valid_models.empty:
            best_mae = valid_models.loc[valid_models['MAE'].idxmin(), 'Model']
            best_rmse = valid_models.loc[valid_models['RMSE'].idxmin(), 'Model']
            
            # Check if Exponential Smoothing was excluded
            overall_best_mae = district_perf.loc[district_perf['MAE'].idxmin(), 'Model']
            overall_best_rmse = district_perf.loc[district_perf['RMSE'].idxmin(), 'Model']
            
            mae_note = " (ES excluded)" if overall_best_mae == 'ExponentialSmoothing' else ""
            rmse_note = " (ES excluded)" if overall_best_rmse == 'ExponentialSmoothing' else ""
            
            rankings.append({
                'District': district,
                'Best MAE': best_mae + mae_note,
                'Best RMSE': best_rmse + rmse_note
            })
        else:
            rankings.append({
                'District': district,
                'Best MAE': 'N/A',
                'Best RMSE': 'N/A'
            })
    
    rankings_df = pd.DataFrame(rankings)
    st.dataframe(rankings_df, width="stretch")
    
    # Model win counts (reliable models only)
    col1, col2 = st.columns(2)
    
    with col1:
        # Clean the model names by removing notes
        clean_mae = [name.split(' (')[0] for name in rankings_df['Best MAE'] if name != 'N/A']
        if clean_mae:
            mae_wins = pd.Series(clean_mae).value_counts()
            fig_mae_wins = px.pie(values=mae_wins.values, names=mae_wins.index, 
                                 title="Best MAE Model Distribution (Reliable Models)")
            st.plotly_chart(fig_mae_wins, config={"responsive": True})
        else:
            st.info("No MAE data available for reliable models")
    
    with col2:
        # Clean the model names by removing notes
        clean_rmse = [name.split(' (')[0] for name in rankings_df['Best RMSE'] if name != 'N/A']
        if clean_rmse:
            rmse_wins = pd.Series(clean_rmse).value_counts()
            fig_rmse_wins = px.pie(values=rmse_wins.values, names=rmse_wins.index, 
                                  title="Best RMSE Model Distribution (Reliable Models)")
            st.plotly_chart(fig_rmse_wins, config={"responsive": True})
        else:
            st.info("No RMSE data available for reliable models")

def show_data_explorer_page(data, predictions, performance):
    """Show data explorer page"""
    st.header("📋 Data Explorer")
    
    # Tabs for different data views
    tab1, tab2, tab3 = st.tabs(["📊 Historical Data", "🔮 Predictions", "📈 Performance"])
    
    with tab1:
        st.subheader("Historical Dengue Data (2010-2025)")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_districts = st.multiselect(
                "Select Districts", 
                options=sorted(data['District'].unique()),
                default=sorted(data['District'].unique())[:5]
            )
        
        with col2:
            year_range = st.slider(
                "Select Year Range",
                min_value=2010,
                max_value=2025,
                value=(2010, 2025)
            )
        
        # Filter data
        filtered_data = data[
            (data['District'].isin(selected_districts)) &
            (data['Date'].dt.year >= year_range[0]) &
            (data['Date'].dt.year <= year_range[1])
        ]
        
        st.dataframe(filtered_data, width="stretch")
        
        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_dengue_data.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("2026 Predictions")
        st.dataframe(predictions, width="stretch")
        
        # Download button
        csv_pred = predictions.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv_pred,
            file_name="dengue_predictions_2026.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Model Performance Metrics")
        st.dataframe(performance, width="stretch")
        
        # Download button
        csv_perf = performance.to_csv(index=False)
        st.download_button(
            label="📥 Download Performance Data",
            data=csv_perf,
            file_name="model_performance.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()