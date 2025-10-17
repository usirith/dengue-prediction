#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

# ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Prophet for time series forecasting
from prophet import Prophet

class DengueTimeSeriesPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.models = {}
        self.predictions = {}
        self.model_performance = {}
        
    def load_and_clean_data(self):
        print("Loading and cleaning data...")
        
        self.data = pd.read_csv(self.data_path)
        
        self.data = self.data.drop(columns=[col for col in self.data.columns if 'Unnamed' in col or col == ''])
        
        self.data.columns = ['Date', 'District', 'Cases']
        
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        self.data = self.data.dropna()
        
        self.data = self.data.sort_values(['District', 'Date'])
        
        print(f"Data shape: {self.data.shape}")
        print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"Districts: {self.data['District'].nunique()}")
        print(f"Unique districts: {sorted(self.data['District'].unique())}")
        
        return self.data
    
    def explore_data(self):
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print("\nBasic Statistics:")
        print(self.data['Cases'].describe())
        
        district_stats = self.data.groupby('District')['Cases'].agg(['mean', 'std', 'min', 'max', 'sum'])
        print(f"\nTop 5 districts by average cases:")
        print(district_stats.sort_values('mean', ascending=False).head())
        
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        
        yearly_cases = self.data.groupby('Year')['Cases'].sum()
        monthly_cases = self.data.groupby('Month')['Cases'].mean()
        
        print(f"\nYearly trend:")
        print(yearly_cases)
        
        print(f"\nSeasonal pattern (average by month):")
        print(monthly_cases)
        
    def create_features(self, district_data):
        df = district_data.copy()
        df = df.sort_values('Date')
        
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df['Cases'].shift(lag)
        
        for window in [3, 6, 12]:
            df[f'rolling_mean_{window}'] = df['Cases'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['Cases'].rolling(window=window).std()
        
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['year'] = df['Date'].dt.year
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df['trend'] = range(len(df))
        
        return df
    
    def check_stationarity(self, series, district_name):
        result = adfuller(series.dropna())
        
        print(f"\nStationarity test for {district_name}:")
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        
        if result[1] <= 0.05:
            print("Series is stationary")
            return True
        else:
            print("Series is non-stationary")
            return False
    
    def fit_arima_model(self, district_data, district_name):
        try:
            ts_data = district_data.set_index('Date')['Cases']
            
            is_stationary = self.check_stationarity(ts_data, district_name)
            
            best_aic = float('inf')
            best_params = (1, 1, 1)
            
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(ts_data, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                        except:
                            continue
            
            model = ARIMA(ts_data, order=best_params)
            fitted_model = model.fit()
            
            forecast = fitted_model.forecast(steps=12)
            forecast_dates = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
            
            return {
                'model': fitted_model,
                'forecast': forecast,
                'forecast_dates': forecast_dates,
                'params': best_params,
                'aic': best_aic
            }
            
        except Exception as e:
            print(f"ARIMA failed for {district_name}: {str(e)}")
            return None
    
    # def fit_exponential_smoothing(self, district_data, district_name):
    #     try:
    #         ts_data = district_data.set_index('Date')['Cases']
            
    #         model = ExponentialSmoothing(
    #             ts_data, 
    #             trend='add', 
    #             seasonal='add', 
    #             seasonal_periods=12
    #         )
    #         fitted_model = model.fit()
            
    #         forecast = fitted_model.forecast(steps=12)
    #         forecast_dates = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
            
    #         return {
    #             'model': fitted_model,
    #             'forecast': forecast,
    #             'forecast_dates': forecast_dates
    #         }
            
    #     except Exception as e:
    #         print(f"Exponential Smoothing failed for {district_name}: {str(e)}")
    #         return None
    
    def fit_random_forest(self, district_data, district_name):
        try:
            df_features = self.create_features(district_data)
            
            df_features = df_features.dropna()
            
            if len(df_features) < 24: 
                print(f"Insufficient data for Random Forest for {district_name}")
                return None
            
            feature_cols = [col for col in df_features.columns if col not in ['Date', 'Cases', 'District']]
            X = df_features[feature_cols]
            y = df_features['Cases']
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            rf_model.fit(X, y)
            
            last_date = district_data['Date'].max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
            
            future_features = []
            for i, date in enumerate(future_dates):
                feature_dict = {
                    'month': date.month,
                    'quarter': date.quarter,
                    'year': date.year,
                    'month_sin': np.sin(2 * np.pi * date.month / 12),
                    'month_cos': np.cos(2 * np.pi * date.month / 12),
                    'trend': len(district_data) + i
                }
                
                last_cases = district_data['Cases'].iloc[-12:].values
                feature_dict['lag_1'] = last_cases[-1] if len(last_cases) >= 1 else district_data['Cases'].mean()
                feature_dict['lag_2'] = last_cases[-2] if len(last_cases) >= 2 else district_data['Cases'].mean()
                feature_dict['lag_3'] = last_cases[-3] if len(last_cases) >= 3 else district_data['Cases'].mean()
                feature_dict['lag_6'] = last_cases[-6] if len(last_cases) >= 6 else district_data['Cases'].mean()
                feature_dict['lag_12'] = last_cases[-12] if len(last_cases) >= 12 else district_data['Cases'].mean()
                
                feature_dict['rolling_mean_3'] = np.mean(last_cases[-3:]) if len(last_cases) >= 3 else district_data['Cases'].mean()
                feature_dict['rolling_mean_6'] = np.mean(last_cases[-6:]) if len(last_cases) >= 6 else district_data['Cases'].mean()
                feature_dict['rolling_mean_12'] = np.mean(last_cases[-12:]) if len(last_cases) >= 12 else district_data['Cases'].mean()
                feature_dict['rolling_std_3'] = np.std(last_cases[-3:]) if len(last_cases) >= 3 else district_data['Cases'].std()
                feature_dict['rolling_std_6'] = np.std(last_cases[-6:]) if len(last_cases) >= 6 else district_data['Cases'].std()
                feature_dict['rolling_std_12'] = np.std(last_cases[-12:]) if len(last_cases) >= 12 else district_data['Cases'].std()
                
                future_features.append(feature_dict)
            
            future_df = pd.DataFrame(future_features)
            
            for col in feature_cols:
                if col not in future_df.columns:
                    future_df[col] = 0
            
            future_df = future_df[feature_cols]  
            
            forecast = rf_model.predict(future_df)
            
            return {
                'model': rf_model,
                'forecast': forecast,
                'forecast_dates': future_dates,
                'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_))
            }
            
        except Exception as e:
            print(f"Random Forest failed for {district_name}: {str(e)}")
            return None
    
    def fit_prophet_model(self, district_data, district_name):
        try:
            prophet_data = district_data[['Date', 'Cases']].copy()
            prophet_data.columns = ['ds', 'y']
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model.fit(prophet_data)
            
            future = model.make_future_dataframe(periods=12, freq='MS')
            
            forecast = model.predict(future)
            
            future_forecast = forecast.tail(12)
            
            return {
                'model': model,
                'forecast': future_forecast['yhat'].values,
                'forecast_dates': future_forecast['ds'].values,
                'full_forecast': forecast
            }
            
        except Exception as e:
            print(f"Prophet failed for {district_name}: {str(e)}")
            return None
    
    def train_models(self):
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        districts = self.data['District'].unique()
        
        for district in districts:
            print(f"\nTraining models for {district}...")
            
            district_data = self.data[self.data['District'] == district].copy()
            
            self.models[district] = {}
            
            print(f"  Fitting ARIMA...")
            arima_result = self.fit_arima_model(district_data, district)
            if arima_result:
                self.models[district]['ARIMA'] = arima_result
            
            print(f"  Fitting Exponential Smoothing...")
            es_result = self.fit_exponential_smoothing(district_data, district)
            if es_result:
                self.models[district]['ExponentialSmoothing'] = es_result
            
            print(f"  Fitting Random Forest...")
            rf_result = self.fit_random_forest(district_data, district)
            if rf_result:
                self.models[district]['RandomForest'] = rf_result
            
            print(f"  Fitting Prophet...")
            prophet_result = self.fit_prophet_model(district_data, district)
            if prophet_result:
                self.models[district]['Prophet'] = prophet_result
    
    def evaluate_models(self):
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        for district in self.models.keys():
            print(f"\nEvaluating models for {district}...")
            
            district_data = self.data[self.data['District'] == district].copy()
            
            if len(district_data) >= 24:
                train_data = district_data.iloc[:-12]
                test_data = district_data.iloc[-12:]
                
                self.model_performance[district] = {}
                
                for model_name, model_info in self.models[district].items():
                    try:
                        if model_name == 'ARIMA':
                            ts_train = train_data.set_index('Date')['Cases']
                            arima_model = ARIMA(ts_train, order=model_info['params'])
                            fitted = arima_model.fit()
                            pred = fitted.forecast(steps=12)
                        
                        elif model_name == 'ExponentialSmoothing':
                            ts_train = train_data.set_index('Date')['Cases']
                            es_model = ExponentialSmoothing(ts_train, trend='add', seasonal='add', seasonal_periods=12)
                            fitted = es_model.fit()
                            pred = fitted.forecast(steps=12)
                        
                        elif model_name == 'RandomForest':
                            pred = model_info['forecast'][:12]  
                        
                        elif model_name == 'Prophet':
                            prophet_data = train_data[['Date', 'Cases']].copy()
                            prophet_data.columns = ['ds', 'y']
                            prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                            prophet_model.fit(prophet_data)
                            future = prophet_model.make_future_dataframe(periods=12, freq='MS')
                            forecast = prophet_model.predict(future)
                            pred = forecast.tail(12)['yhat'].values
                        
                        actual = test_data['Cases'].values
                        
                        mae = mean_absolute_error(actual, pred)
                        mse = mean_squared_error(actual, pred)
                        rmse = np.sqrt(mse)
                        
                        self.model_performance[district][model_name] = {
                            'MAE': mae,
                            'MSE': mse,
                            'RMSE': rmse,
                            'predictions': pred,
                            'actual': actual
                        }
                        
                        print(f"  {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}")
                        
                    except Exception as e:
                        print(f"  {model_name}: Evaluation failed - {str(e)}")
    
    def generate_predictions(self):
        print("\n" + "="*50)
        print("GENERATING 2026 PREDICTIONS")
        print("="*50)
        
        prediction_results = []
        
        for district in self.models.keys():
            print(f"\nPredictions for {district}:")
            
            district_predictions = {'District': district}
            
            for model_name, model_info in self.models[district].items():
                forecast = model_info['forecast']
                dates = model_info['forecast_dates']
                
                for i, (date, pred) in enumerate(zip(dates, forecast)):
                    if hasattr(date, 'strftime'):
                        month_name = f"{date.strftime('%Y-%m')}"
                    else:
                        date_pd = pd.to_datetime(date)
                        month_name = f"{date_pd.strftime('%Y-%m')}"
                    if f'{model_name}_{month_name}' not in district_predictions:
                        district_predictions[f'{model_name}_{month_name}'] = pred
                
                annual_total = np.sum(forecast)
                district_predictions[f'{model_name}_Annual_Total'] = annual_total
                
                print(f"  {model_name}: Annual total = {annual_total:.0f} cases")
            
            prediction_results.append(district_predictions)
        
        self.predictions_df = pd.DataFrame(prediction_results)
        
        return self.predictions_df
    
    def create_visualizations(self):
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)
        
        plt.figure(figsize=fig_size)
        
        top_districts = self.data.groupby('District')['Cases'].sum().nlargest(6).index
        
        for i, district in enumerate(top_districts):
            plt.subplot(2, 3, i+1)
            district_data = self.data[self.data['District'] == district]
            plt.plot(district_data['Date'], district_data['Cases'])
            plt.title(f'{district}')
            plt.xticks(rotation=45)
            plt.ylabel('Cases')
        
        plt.tight_layout()
        plt.savefig('/home/navin/CODE/ml/historical_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if hasattr(self, 'predictions_df'):
            plt.figure(figsize=(20, 12))
            
            model_names = ['ARIMA', 'ExponentialSmoothing', 'RandomForest', 'Prophet']
            
            for i, model in enumerate(model_names):
                plt.subplot(2, 2, i+1)
                
                annual_cols = [col for col in self.predictions_df.columns if f'{model}_Annual_Total' in col]
                if annual_cols:
                    values = self.predictions_df[annual_cols[0]].values
                    districts = self.predictions_df['District'].values
                    
                    sorted_idx = np.argsort(values)[::-1]
                    
                    plt.barh(range(len(values)), values[sorted_idx])
                    plt.yticks(range(len(values)), [districts[idx] for idx in sorted_idx])
                    plt.xlabel('Predicted Annual Cases')
                    plt.title(f'{model} - 2026 Annual Predictions')
                    plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.savefig('/home/navin/CODE/ml/predictions_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_results(self):
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        
        if hasattr(self, 'predictions_df'):
            self.predictions_df.to_csv('/home/navin/CODE/ml/dengue_predictions_2026.csv', index=False)
            print("Predictions saved to: dengue_predictions_2026.csv")
        
        if hasattr(self, 'model_performance'):
            performance_data = []
            for district, models in self.model_performance.items():
                for model_name, metrics in models.items():
                    performance_data.append({
                        'District': district,
                        'Model': model_name,
                        'MAE': metrics['MAE'],
                        'MSE': metrics['MSE'],
                        'RMSE': metrics['RMSE']
                    })
            
            performance_df = pd.DataFrame(performance_data)
            performance_df.to_csv('/home/navin/CODE/ml/model_performance.csv', index=False)
            print("Model performance saved to: model_performance.csv")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("DENGUE TIME SERIES PREDICTION ANALYSIS")
        print("="*60)
        
        self.load_and_clean_data()
        self.explore_data()
        
        self.train_models()
        
        self.evaluate_models()
        
        self.generate_predictions()
        
        self.create_visualizations()
        
        self.save_results()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)

def main():
    predictor = DengueTimeSeriesPredictor('/home/navin/CODE/ml/Dengue_Data (2010-2025).csv')
    
    predictor.run_complete_analysis()
    
    return predictor

if __name__ == "__main__":
    predictor = main()
