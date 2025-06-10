# Solar Power Generation Predictor ☀️

A high-precision solar power prediction system built with Streamlit that uses ensemble machine learning to predict solar power generation with comprehensive environmental parameters and confidence intervals.

## Features

### 🎯 Core Functionality
- **Ensemble Machine Learning**: Combines XGBoost, Random Forest, and Gradient Boosting models
- **High-Precision Predictions**: Advanced feature engineering with 20+ environmental parameters
- **Confidence Intervals**: Bootstrap-based uncertainty estimation
- **Location-Based Intelligence**: Automatic parameter estimation from similar locations
- **Interactive Visualization**: Geographic mapping and performance metrics

### 🔬 Advanced Features
- Solar angle calculations (declination, optimal tilt)
- Air mass coefficient calculations
- Temperature efficiency factors
- Seasonal adjustments
- Climate interaction modeling
- Power density analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Packages
```bash
pip install streamlit pandas numpy scikit-learn xgboost matplotlib seaborn geopandas contextily shapely joblib
```

### Installation Steps
1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the dataset file: `improved_solar_data.txt`
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Data Requirements

### Primary Dataset
The application expects a file named `improved_solar_data.txt` with the following columns:
- `Location`: Geographic location name
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `output`: Solar power output (MW)
- `number_of_panels_used`: Number of solar panels
- `GHI (kWh/m²/year)`: Global Horizontal Irradiance
- `Elevation (m)`: Elevation above sea level
- `Avg Temp (°C)`: Average temperature
- `Cloud Cover (%)`: Cloud cover percentage
- `Air Pollution Index`: Categorical (Low/Moderate/High)

### Model Persistence
The app automatically saves trained models as:
- `ensemble_models.pkl`: Trained ML models
- `feature_scaler.pkl`: Feature scaling parameters
- `model_features.pkl`: Feature list for predictions

## Usage

### Basic Operation
1. **Launch the app**: Run `streamlit run app.py`
2. **Set location**: Input latitude and longitude coordinates
3. **Configure system**: Specify number of solar panels
4. **Environmental parameters**: Adjust climate and environmental factors
5. **Generate prediction**: Click the prediction button for results

### Input Parameters

#### Location Settings
- **Latitude**: -90° to 90° (decimal degrees)
- **Longitude**: -180° to 180° (decimal degrees)

#### System Configuration
- **Number of Panels**: 1,000 to 50,000,000 panels

#### Environmental Parameters
- **Global Horizontal Irradiance**: 800-2,500 kWh/m²/year
- **Elevation**: 0-5,000 meters
- **Average Temperature**: -10°C to 50°C
- **Cloud Cover**: 0-100%
- **Air Pollution Level**: Low/Moderate/High

#### Advanced Options
- **Day of Year**: 1-365 (for solar angle calculations)
- **Confidence Intervals**: Enable/disable uncertainty estimation
- **Model Breakdown**: Show individual model predictions

### Output Interpretation

#### Main Prediction
- **Predicted Output**: Primary power generation forecast (MW)
- **Confidence Interval**: 95% uncertainty range
- **Uncertainty Range**: ±MW variation

#### Performance Metrics
- **Power per Panel**: Individual panel efficiency (kW)
- **Annual Generation**: Yearly power output (GWh)
- **Capacity Factor**: Operational efficiency percentage

#### Solar Potential Assessment
Five-tier rating system:
- ⭐⭐⭐⭐⭐ Excellent (80-100%)
- ⭐⭐⭐⭐ Very Good (65-79%)
- ⭐⭐⭐ Good (50-64%)
- ⭐⭐ Moderate (35-49%)
- ⭐ Limited (0-34%)

## Technical Architecture

### Machine Learning Models
1. **XGBoost Regressor** (50% weight)
   - 1000 estimators, 0.05 learning rate
   - Regularization and early stopping
   
2. **Random Forest** (30% weight)
   - 500 trees with depth optimization
   - Feature importance analysis
   
3. **Gradient Boosting** (20% weight)
   - 500 estimators with subsample optimization
   - Robust to outliers

### Feature Engineering
The system creates 20+ engineered features including:

#### Geographic Features
- Distance from equator
- Coastal proximity indicators
- Solar declination angles

#### Climate Features
- Temperature efficiency factors
- Clear sky index
- Air mass coefficients

#### Interaction Features
- GHI-temperature interactions
- Elevation-temperature correlations
- Latitude-irradiance combinations

### Uncertainty Quantification
- Bootstrap sampling (50 iterations)
- Gaussian noise injection (2% variance)
- Percentile-based confidence intervals

## Configuration

### Environment Variables
No environment variables required for basic operation.

### Customization Options
- Modify ensemble weights in `predict_with_ensemble()`
- Adjust bootstrap iterations in `calculate_confidence_interval()`
- Update model hyperparameters in `create_ensemble_model()`

## Performance Optimization

### Model Training
- First run trains models (2-5 minutes)
- Subsequent runs load pre-trained models (seconds)
- Models automatically saved for persistence

### Memory Usage
- Robust scaler reduces memory footprint
- Feature selection optimizes performance
- Error handling prevents crashes

## Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
# Install missing packages
pip install package_name
```

#### Dataset Not Found
Ensure `improved_solar_data.txt` is in the same directory as `app.py`

#### Model Training Fails
Check data format and column names match requirements

#### Prediction Errors
Verify input parameters are within valid ranges

### Error Messages
- **"No models were successfully trained!"**: Check dataset format
- **"All model predictions failed!"**: Verify input data consistency
- **Model loading warnings**: Normal for first-time training

## Contributors

### Project Owner
- **Akshat Sharma** - [@Akshat-Sharma-110011](https://github.com/Akshat-Sharma-110011)
  - *Project Creator & Lead Developer*
  - Designed and implemented the ensemble machine learning architecture
  - Developed advanced feature engineering pipeline
  - Created the Streamlit user interface

### Core Contributors
- **[Contributor Name]** - [@github-username](https://github.com/github-username)
  - *Role/Contribution*
  - Contribution description

- **[Contributor Name]** - [@github-username](https://github.com/github-username)
  - *Role/Contribution*
  - Contribution description

- **[Contributor Name]** - [@github-username](https://github.com/github-username)
  - *Role/Contribution*
  - Contribution description

### How to Contribute

#### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

#### Code Style
- Follow PEP 8 conventions
- Add docstrings for new functions
- Include error handling for robust operation

#### Contribution Guidelines
- Test all changes thoroughly before submitting
- Update documentation for new features
- Maintain backward compatibility when possible
- Include unit tests for new functionality

## License

This project is open source. Please check the license file for specific terms.

## Support

For issues, questions, or contributions:
1. Check existing documentation
2. Review troubleshooting section
3. Create detailed issue reports
4. Include error messages and system information

## Version History

### Current Version
- Ensemble machine learning implementation
- Advanced feature engineering
- Confidence interval estimation
- Geographic visualization
- Optimization recommendations

### Future Enhancements
- Weather API integration
- Real-time satellite data
- Multiple location batch processing
- Export functionality for reports
- Integration with solar databases

---

**Note**: This application is designed for educational and research purposes. For commercial solar installations, consult with professional solar engineers and use certified design software.
