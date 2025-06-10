import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import math
import warnings
warnings.filterwarnings('ignore')

# Advanced feature engineering functions
def calculate_solar_angles(lat, day_of_year=172):  # Default to summer solstice
    """Calculate solar declination and optimal tilt angle"""
    lat_rad = math.radians(lat)
    declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    declination_rad = math.radians(declination)
    
    # Solar noon angle
    solar_noon_angle = 90 - abs(lat - declination)
    
    # Optimal tilt angle (simplified)
    optimal_tilt = abs(lat) - 10 if abs(lat) > 10 else 0
    
    return declination, solar_noon_angle, optimal_tilt

def calculate_air_mass(lat, elevation):
    """Calculate air mass coefficient based on location"""
    # Simplified air mass calculation
    zenith_angle = abs(lat) * 0.5  # Rough approximation
    air_mass = 1 / (math.cos(math.radians(zenith_angle)) + 0.50572 * (96.07995 - zenith_angle)**(-1.6364))
    
    # Altitude correction
    pressure_ratio = math.exp(-elevation / 8400)  # Barometric formula
    air_mass_corrected = air_mass * pressure_ratio
    
    return air_mass_corrected

def estimate_parameters_from_location(lat, lon, df):
    """Estimate parameters based on similar locations in the dataset"""
    # Find closest locations
    df['distance'] = np.sqrt((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)
    closest_locations = df.nsmallest(5, 'distance')
    
    # Weighted average based on distance
    weights = 1 / (closest_locations['distance'] + 0.001)  # Avoid division by zero
    weights = weights / weights.sum()
    
    estimates = {
        'ghi': (closest_locations['GHI (kWh/m²/year)'] * weights).sum(),
        'elevation': (closest_locations['Elevation (m)'] * weights).sum(),
        'temp': (closest_locations['Avg Temp (°C)'] * weights).sum(),
        'cloud_cover': (closest_locations['Cloud Cover (%)'] * weights).sum(),
        'air_pollution': closest_locations['Air Pollution Index'].mode().iloc[0] if len(closest_locations) > 0 else 'Moderate'
    }
    
    return estimates

# Load and preprocess data with advanced features
@st.cache_data
def load_and_process_data():
    """Load data and create advanced features"""
    df = pd.read_csv("improved_solar_data.txt")
    
    # Basic feature engineering
    df['Panel_Efficiency'] = df['output'] * 1000 / (df['GHI (kWh/m²/year)'] * df['number_of_panels_used'])
    df['Country'] = df['Location'].apply(lambda x: x.split(',')[-1].strip())
    
    # Advanced feature engineering
    df['Solar_Declination'], df['Solar_Noon_Angle'], df['Optimal_Tilt'] = zip(*df.apply(
        lambda row: calculate_solar_angles(row['latitude']), axis=1))
    
    df['Air_Mass'] = df.apply(lambda row: calculate_air_mass(row['latitude'], row['Elevation (m)']), axis=1)
    
    # Geographic features
    df['Distance_from_Equator'] = abs(df['latitude'])
    df['Coastal_Proximity'] = df.apply(lambda row: 1 if abs(row['longitude']) > 50 else 0, axis=1)  # Simplified
    
    # Climate features
    df['Temperature_Efficiency_Factor'] = 1 - (df['Avg Temp (°C)'] - 25) * 0.004  # Temperature coefficient
    df['Clear_Sky_Index'] = (100 - df['Cloud Cover (%)']) / 100
    
    # Interaction features
    df['GHI_Temperature_Interaction'] = df['GHI (kWh/m²/year)'] * df['Temperature_Efficiency_Factor']
    df['Elevation_Temp_Interaction'] = df['Elevation (m)'] * df['Avg Temp (°C)']
    df['Latitude_GHI_Interaction'] = abs(df['latitude']) * df['GHI (kWh/m²/year)']
    
    # Power density features
    df['Power_Density'] = df['output'] / df['number_of_panels_used'] * 1000  # kW per panel
    df['Area_Efficiency'] = df['output'] / (df['number_of_panels_used'] * 2)  # Assuming 2m² per panel
    
    # Seasonal adjustments
    df['Seasonal_Factor'] = 1 + 0.2 * np.cos(2 * np.pi * df['latitude'] / 180)  # Simplified seasonal variation
    
    # Convert categorical variables
    pollution_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    df['Air_Pollution_Num'] = df['Air Pollution Index'].map(pollution_map)
    
    # Load world map
    try:
        world = gpd.read_file("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson")
    except:
        world = None
    
    return df, world

def create_ensemble_model(df):
    """Create and train ensemble model with advanced features"""
    
    # Define comprehensive feature set
    features = [
        # Basic features
        'latitude', 'longitude', 'number_of_panels_used',
        'GHI (kWh/m²/year)', 'Elevation (m)', 'Avg Temp (°C)', 'Cloud Cover (%)',
        'Air_Pollution_Num',
        
        # Advanced features
        'Solar_Declination', 'Solar_Noon_Angle', 'Optimal_Tilt', 'Air_Mass',
        'Distance_from_Equator', 'Coastal_Proximity', 'Temperature_Efficiency_Factor',
        'Clear_Sky_Index', 'GHI_Temperature_Interaction', 'Elevation_Temp_Interaction',
        'Latitude_GHI_Interaction', 'Seasonal_Factor'
    ]
    
    # Handle missing values
    df[features] = df[features].fillna(df[features].median())
    
    X = df[features]
    y = df['output']
    
    # Split data with proper stratification handling
    try:
        # Try stratified split first
        y_binned = pd.cut(y, bins=3, labels=False)  # Reduced bins for better distribution
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_binned)
    except ValueError:
        # Fall back to regular split if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'xgb': XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=50
        ),
        'rf': RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    # Train models with error handling
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        try:
            if name == 'xgb':
                # XGBoost with validation set
                model.fit(X_train, y_train, 
                         eval_set=[(X_test, y_test)], 
                         verbose=False)
            else:
                # Other models with scaled features
                X_train_to_use = X_train_scaled if name != 'xgb' else X_train
                model.fit(X_train_to_use, y_train)
            
            trained_models[name] = model
            
            # Evaluate model
            X_test_to_use = X_test_scaled if name != 'xgb' else X_test
            y_pred = model.predict(X_test_to_use)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            model_scores[name] = {'r2': r2, 'mae': mae}
            
        except Exception as e:
            st.warning(f"Model {name} failed to train: {str(e)}")
            continue
    
    # Create ensemble predictions with error handling
    if len(trained_models) == 0:
        raise ValueError("No models were successfully trained!")
    
    ensemble_pred = np.zeros(len(y_test))
    weights = {'xgb': 0.5, 'rf': 0.3, 'gb': 0.2}
    total_weight = 0
    
    for name, weight in weights.items():
        if name in trained_models:
            X_test_to_use = X_test_scaled if name != 'xgb' else X_test
            pred = trained_models[name].predict(X_test_to_use)
            ensemble_pred += weight * pred
            total_weight += weight
    
    # Normalize if not all models were trained
    if total_weight < 1.0:
        ensemble_pred = ensemble_pred / total_weight
    
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    return trained_models, scaler, features, model_scores, ensemble_r2, ensemble_mae

def predict_with_ensemble(models, scaler, features, input_data):
    """Make predictions using ensemble model with error handling"""
    weights = {'xgb': 0.5, 'rf': 0.3, 'gb': 0.2}
    
    predictions = {}
    ensemble_pred = 0
    total_weight = 0
    
    for name, weight in weights.items():
        if name in models:
            try:
                if name == 'xgb':
                    pred = models[name].predict(input_data[features])[0]
                else:
                    pred = models[name].predict(scaler.transform(input_data[features]))[0]
                
                predictions[name] = pred
                ensemble_pred += weight * pred
                total_weight += weight
            except Exception as e:
                st.warning(f"Prediction failed for model {name}: {str(e)}")
                predictions[name] = 0
    
    # Normalize if not all models made predictions
    if total_weight > 0:
        ensemble_pred = ensemble_pred / total_weight
    else:
        ensemble_pred = 0
        st.error("All model predictions failed!")
    
    return ensemble_pred, predictions

def calculate_confidence_interval(models, scaler, features, input_data, n_bootstrap=50):
    """Calculate prediction confidence interval using bootstrap with error handling"""
    predictions = []
    
    for _ in range(n_bootstrap):
        try:
            # Add small random noise to simulate uncertainty
            noisy_input = input_data.copy()
            for feature in features:
                if feature in noisy_input.columns:
                    current_value = noisy_input[feature].iloc[0]
                    if pd.notna(current_value) and current_value != 0:
                        noise = np.random.normal(0, 0.02 * abs(current_value))  # Reduced noise
                        noisy_input[feature] += noise
            
            pred, _ = predict_with_ensemble(models, scaler, features, noisy_input)
            if pred > 0:  # Only include valid predictions
                predictions.append(pred)
        except Exception:
            continue
    
    if len(predictions) < 10:  # If too few valid predictions
        return None, None
    
    predictions = np.array(predictions)
    lower_bound = np.percentile(predictions, 2.5)
    upper_bound = np.percentile(predictions, 97.5)
    
    return lower_bound, upper_bound

# Main Streamlit app
def main():
    st.set_page_config(page_title="Solar Power Predictor", layout="wide", page_icon="☀️")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("Solar Power Generation Predictor")
    st.markdown("""
    **High-Precision Solar Power Prediction System**
    
    This advanced model uses ensemble machine learning with comprehensive environmental parameters 
    to provide highly accurate solar power generation predictions with confidence intervals.
    """)
    
    # Load data and train model
    with st.spinner("Loading data and training advanced models..."):
        df, world = load_and_process_data()
        
        # Check if models exist
        try:
            models = joblib.load('ensemble_models.pkl')
            scaler = joblib.load('feature_scaler.pkl')
            features = joblib.load('model_features.pkl')
            st.success("Pre-trained models loaded successfully!")
            model_scores = {}  # Initialize empty scores for pre-trained models
        except (FileNotFoundError, Exception):
            st.info("Training new ensemble models... This may take a moment.")
            models, scaler, features, model_scores, ensemble_r2, ensemble_mae = create_ensemble_model(df)
            
            # Save models
            try:
                joblib.dump(models, 'ensemble_models.pkl')
                joblib.dump(scaler, 'feature_scaler.pkl')
                joblib.dump(features, 'model_features.pkl')
                st.success(f"Ensemble model trained! R² Score: {ensemble_r2:.4f}, MAE: {ensemble_mae:.2f} MW")
            except Exception as e:
                st.warning(f"Could not save models: {str(e)}")
    
    # Sidebar for inputs
    st.sidebar.header("Prediction Parameters")
    
    # Location inputs
    st.sidebar.subheader("Location")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.0, step=0.1)
    with col2:
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.0, step=0.1)
    
    # Get location-based estimates
    estimates = estimate_parameters_from_location(lat, lon, df)
    
    # System parameters
    st.sidebar.subheader("System Configuration")
    panels = st.sidebar.number_input("Number of Panels", min_value=1000, max_value=50000000, value=100000, step=1000)
    
    # Environmental parameters with smart defaults
    st.sidebar.subheader("Environmental Parameters")
    
    use_estimates = st.sidebar.checkbox("Use location-based estimates", value=True)
    
    if use_estimates:
        ghi = st.sidebar.slider("Global Horizontal Irradiance (kWh/m²/year)", 
                               min_value=800, max_value=2500, value=int(estimates['ghi']), step=50)
        elevation = st.sidebar.slider("Elevation (m)", 
                                     min_value=0, max_value=5000, value=int(estimates['elevation']), step=50)
        avg_temp = st.sidebar.slider("Average Temperature (°C)", 
                                    min_value=-10.0, max_value=50.0, value=estimates['temp'], step=0.5)
        cloud_cover = st.sidebar.slider("Cloud Cover (%)", 
                                       min_value=0, max_value=100, value=int(estimates['cloud_cover']), step=5)
        air_pollution = st.sidebar.selectbox("Air Pollution Level", 
                                            options=['Low', 'Moderate', 'High'], 
                                            index=['Low', 'Moderate', 'High'].index(estimates['air_pollution']))
    else:
        ghi = st.sidebar.slider("Global Horizontal Irradiance (kWh/m²/year)", 
                               min_value=800, max_value=2500, value=1800, step=50)
        elevation = st.sidebar.slider("Elevation (m)", 
                                     min_value=0, max_value=5000, value=200, step=50)
        avg_temp = st.sidebar.slider("Average Temperature (°C)", 
                                    min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
        cloud_cover = st.sidebar.slider("Cloud Cover (%)", 
                                       min_value=0, max_value=100, value=30, step=5)
        air_pollution = st.sidebar.selectbox("Air Pollution Level", 
                                            options=['Low', 'Moderate', 'High'], 
                                            index=1)
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        day_of_year = st.slider("Day of Year (for solar angle calculation)", 
                               min_value=1, max_value=365, value=172, step=1)
        show_confidence = st.checkbox("Show Confidence Intervals", value=True)
        show_breakdown = st.checkbox("Show Model Breakdown", value=True)
    
    # Create prediction button
    if st.sidebar.button("Generate Prediction", type="primary"):
        # Prepare input data
        input_data = prepare_input_data(lat, lon, panels, ghi, elevation, avg_temp, 
                                       cloud_cover, air_pollution, day_of_year)
        
        # Make prediction
        with st.spinner("Generating high-precision prediction..."):
            prediction, individual_predictions = predict_with_ensemble(models, scaler, features, input_data)
            
            if show_confidence:
                lower_bound, upper_bound = calculate_confidence_interval(models, scaler, features, input_data)
            else:
                lower_bound, upper_bound = None, None
            
        # Display results
        display_results(prediction, individual_predictions, input_data, lat, lon, 
                       lower_bound, upper_bound, show_breakdown, world)

def prepare_input_data(lat, lon, panels, ghi, elevation, avg_temp, cloud_cover, air_pollution, day_of_year):
    """Prepare input data with all engineered features"""
    
    # Calculate advanced features
    declination, solar_noon_angle, optimal_tilt = calculate_solar_angles(lat, day_of_year)
    air_mass = calculate_air_mass(lat, elevation)
    
    # Convert air pollution to numerical
    pollution_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    air_pollution_num = pollution_map[air_pollution]
    
    # Calculate derived features
    distance_from_equator = abs(lat)
    coastal_proximity = 1 if abs(lon) > 50 else 0
    temperature_efficiency_factor = 1 - (avg_temp - 25) * 0.004
    clear_sky_index = (100 - cloud_cover) / 100
    
    # Interaction features
    ghi_temperature_interaction = ghi * temperature_efficiency_factor
    elevation_temp_interaction = elevation * avg_temp
    latitude_ghi_interaction = abs(lat) * ghi
    seasonal_factor = 1 + 0.2 * math.cos(2 * math.pi * lat / 180)
    
    input_data = pd.DataFrame({
        'latitude': [lat],
        'longitude': [lon],
        'number_of_panels_used': [panels],
        'GHI (kWh/m²/year)': [ghi],
        'Elevation (m)': [elevation],
        'Avg Temp (°C)': [avg_temp],
        'Cloud Cover (%)': [cloud_cover],
        'Air_Pollution_Num': [air_pollution_num],
        'Solar_Declination': [declination],
        'Solar_Noon_Angle': [solar_noon_angle],
        'Optimal_Tilt': [optimal_tilt],
        'Air_Mass': [air_mass],
        'Distance_from_Equator': [distance_from_equator],
        'Coastal_Proximity': [coastal_proximity],
        'Temperature_Efficiency_Factor': [temperature_efficiency_factor],
        'Clear_Sky_Index': [clear_sky_index],
        'GHI_Temperature_Interaction': [ghi_temperature_interaction],
        'Elevation_Temp_Interaction': [elevation_temp_interaction],
        'Latitude_GHI_Interaction': [latitude_ghi_interaction],
        'Seasonal_Factor': [seasonal_factor]
    })
    
    return input_data

def display_results(prediction, individual_predictions, input_data, lat, lon, 
                   lower_bound, upper_bound, show_breakdown, world):
    """Display comprehensive prediction results"""
    
    # Main prediction display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Prediction Results")
        
        # Main prediction with confidence interval
        if lower_bound is not None and upper_bound is not None:
            st.markdown(f"""
            <div class="metric-card">
                <h2>Predicted Output: {prediction:,.2f} MW</h2>
                <p>95% Confidence Interval: {lower_bound:,.2f} - {upper_bound:,.2f} MW</p>
                <p>Uncertainty Range: ±{(upper_bound - lower_bound)/2:,.2f} MW</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h2>Predicted Output: {prediction:,.2f} MW</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance metrics
        panels = input_data['number_of_panels_used'].iloc[0]
        ghi = input_data['GHI (kWh/m²/year)'].iloc[0]
        
        # Calculate efficiency metrics
        power_per_panel = prediction * 1000 / panels  # kW per panel
        capacity_factor = prediction / (panels * 0.4 / 1000)  # Assuming 400W panels
        annual_generation = prediction * 365 * 24 * capacity_factor / 1000  # MWh per year
        annual_generation_gwh = annual_generation / 1000  # Convert MWh to GWh

        col1a, col1b, col1c = st.columns(3)
        with col1a:
            st.metric("Power per Panel", f"{power_per_panel:.3f} kW")
        with col1b:
            st.metric("Annual Generation", f"{annual_generation_gwh:,.2f} GWh")
        with col1c:
            st.metric("Capacity Factor", f"{capacity_factor:.1f}%")
        # Model breakdown
        if show_breakdown:
            st.markdown("### Model Breakdown")
            breakdown_df = pd.DataFrame({
                'Model': ['XGBoost', 'Random Forest', 'Gradient Boosting', 'Ensemble'],
                'Prediction (MW)': [
                    individual_predictions['xgb'],
                    individual_predictions['rf'],
                    individual_predictions['gb'],
                    prediction
                ],
                'Weight': ['50%', '30%', '20%', '100%']
            })
            st.dataframe(breakdown_df, use_container_width=True)
    
    with col2:
        st.markdown("## Analysis")
        
        # Solar potential assessment
        solar_potential = assess_solar_potential(lat, ghi, input_data['Clear_Sky_Index'].iloc[0])
        
        st.markdown(f"""
        <style>
        .info-box h4, .info-box p {{
                    color : #333;
        }}
        </style>
        <div class="info-box">
            <h4>Solar Potential: {solar_potential['level']}</h4>
            <p>{solar_potential['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key factors affecting prediction
        st.markdown("### Key Impact Factors")
        
        factors = [
            ("Global Irradiance", ghi, "kWh/m²/year"),
            ("Temperature Efficiency", input_data['Temperature_Efficiency_Factor'].iloc[0] * 100, "%"),
            ("Clear Sky Index", input_data['Clear_Sky_Index'].iloc[0] * 100, "%"),
            ("Optimal Tilt Angle", input_data['Optimal_Tilt'].iloc[0], "degrees"),
            ("Air Mass Factor", input_data['Air_Mass'].iloc[0], ""),
        ]
        
        for factor, value, unit in factors:
            st.write(f"**{factor}:** {value:.1f} {unit}")
    
    # Location visualization
    if world is not None:
        st.markdown("## Location Visualization")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot world map
        world.plot(ax=ax, alpha=0.4, color="lightgray", edgecolor="white")
        
        # Plot location
        gdf = gpd.GeoDataFrame({'Name': ['Solar Farm Location'], 'geometry': [Point(lon, lat)]}, 
                              crs="EPSG:4326")
        gdf.plot(ax=ax, color="red", markersize=200, edgecolor="darkred", linewidth=2)
        
        # Customize map
        ax.set_xlim(lon - 20, lon + 20)
        ax.set_ylim(lat - 15, lat + 15)
        ax.set_title(f"Solar Farm Location: {lat:.2f}°N, {lon:.2f}°E", fontsize=16, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        st.pyplot(fig)
    
    # Recommendations
    st.markdown("## Optimization Recommendations")
    
    recommendations = generate_recommendations(input_data, prediction, lat)
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")

def assess_solar_potential(lat, ghi, clear_sky_index):
    """Assess solar potential based on location and conditions"""
    
    # Calculate composite score
    lat_score = max(0, 100 - abs(lat) * 2)  # Closer to equator is better
    ghi_score = min(100, (ghi - 800) / 1700 * 100)  # Normalize GHI
    clear_score = clear_sky_index * 100
    
    composite_score = (lat_score + ghi_score + clear_score) / 3
    
    if composite_score >= 80:
        return {
            'level': 'Excellent ⭐⭐⭐⭐⭐',
            'description': 'Outstanding solar conditions with high irradiance and minimal cloud cover.'
        }
    elif composite_score >= 65:
        return {
            'level': 'Very Good ⭐⭐⭐⭐',
            'description': 'Very favorable conditions for solar power generation.'
        }
    elif composite_score >= 50:
        return {
            'level': 'Good ⭐⭐⭐',
            'description': 'Good solar potential with room for optimization.'
        }
    elif composite_score >= 35:
        return {
            'level': 'Moderate ⭐⭐',
            'description': 'Moderate solar potential. Consider efficiency improvements.'
        }
    else:
        return {
            'level': 'Limited ⭐',
            'description': 'Limited solar potential. Significant optimization needed.'
        }

def generate_recommendations(input_data, prediction, lat):
    """Generate optimization recommendations"""
    recommendations = []
    
    # Temperature optimization
    temp = input_data['Avg Temp (°C)'].iloc[0]
    if temp > 30:
        recommendations.append("Consider active cooling systems to improve panel efficiency in high temperatures.")
    elif temp < 10:
        recommendations.append("Cold climate detected. Ensure panels are rated for low temperatures.")
    
    # Tilt angle optimization
    optimal_tilt = input_data['Optimal_Tilt'].iloc[0]
    recommendations.append(f"Optimize panel tilt angle to {optimal_tilt:.1f}° for maximum solar capture.")
    
    # Cloud cover mitigation
    cloud_cover = input_data['Cloud Cover (%)'].iloc[0]
    if cloud_cover > 50:
        recommendations.append("High cloud cover detected. Consider tracking systems to maximize diffuse light capture.")
    
    # Air pollution considerations
    air_pollution = input_data['Air_Pollution_Num'].iloc[0]
    if air_pollution >= 1:
        recommendations.append("Implement regular panel cleaning schedules due to air pollution levels.")
    
    # Elevation benefits
    elevation = input_data['Elevation (m)'].iloc[0]
    if elevation > 1000:
        recommendations.append("High elevation provides cleaner air and better solar irradiance.")
    
    # Seasonal adjustments
    if abs(lat) > 30:
        recommendations.append("Consider seasonal tilt adjustments for locations far from the equator.")
    
    return recommendations

if __name__ == "__main__":
    main()