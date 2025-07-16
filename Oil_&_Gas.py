import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Custom accuracy metrics for volatility prediction
def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate comprehensive accuracy metrics for volatility prediction"""
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {}
    
    # 1. Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    # 2. Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # 3. Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    # 4. R-squared (coefficient of determination)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # 5. Directional Accuracy (for changes in volatility)
    if len(y_true_clean) > 1:
        true_direction = np.diff(y_true_clean) > 0
        pred_direction = np.diff(y_pred_clean) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = 0
    
    # 6. Normalized RMSE (NRMSE) - normalized by mean of true values
    nrmse = rmse / np.mean(y_true_clean) * 100
    
    # 7. Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
    
    # 8. Theil's U statistic (useful for time series)
    theil_u = np.sqrt(np.mean((y_pred_clean - y_true_clean)**2)) / np.sqrt(np.mean(y_true_clean**2))
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'NRMSE': nrmse,
        'SMAPE': smape,
        'Theil_U': theil_u
    }

def print_accuracy_report(metrics, ticker_name):
    """Print a formatted accuracy report"""
    print(f"\n=== {ticker_name} Accuracy Metrics ===")
    print(f"Root Mean Square Error (RMSE):     {metrics['RMSE']:.6f}")
    print(f"Mean Absolute Error (MAE):         {metrics['MAE']:.6f}")
    print(f"Mean Absolute Percentage Error:    {metrics['MAPE']:.2f}%")
    print(f"R-squared (R²):                    {metrics['R2']:.4f}")
    print(f"Directional Accuracy:              {metrics['Directional_Accuracy']:.2f}%")
    print(f"Normalized RMSE:                   {metrics['NRMSE']:.2f}%")
    print(f"Symmetric MAPE:                    {metrics['SMAPE']:.2f}%")
    print(f"Theil's U statistic:               {metrics['Theil_U']:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if metrics['R2'] > 0.7:
        print("- Excellent model performance (R² > 0.7)")
    elif metrics['R2'] > 0.5:
        print("- Good model performance (R² > 0.5)")
    elif metrics['R2'] > 0.3:
        print("- Moderate model performance (R² > 0.3)")
    else:
        print("- Poor model performance (R² < 0.3)")
    
    if metrics['Directional_Accuracy'] > 60:
        print(f"- Good directional prediction (>{metrics['Directional_Accuracy']:.1f}% accuracy)")
    else:
        print(f"- Poor directional prediction ({metrics['Directional_Accuracy']:.1f}% accuracy)")
    
    if metrics['MAPE'] < 10:
        print("- Very accurate predictions (MAPE < 10%)")
    elif metrics['MAPE'] < 20:
        print("- Reasonably accurate predictions (MAPE < 20%)")
    else:
        print("- Less accurate predictions (MAPE > 20%)")

# 1. Data Collection
# Fetch historical data for oil companies and S&P 500
tickers = ["XOM", "CVX", "^GSPC"]  # Exxon, Chevron, S&P 500
start_date = "2015-01-01"
end_date = "2025-07-11"  # Up to today

# Download data for all tickers
data_dict = {}
original_data_dict = {}

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    original_data_dict[ticker] = data.copy()  # Keep original data for price plotting
    data = data[['Close']]  # Use closing prices
    
    # Calculate Volatility (daily returns standard deviation over a rolling window)
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(window=20).std()  # 20-day rolling volatility
    data = data.dropna()
    data_dict[ticker] = data

# 2. Function to create LSTM model and predict volatility
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def build_and_train_model(volatility_data, seq_length=60):
    # Prepare Data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(volatility_data.values.reshape(-1, 1))
    
    # Create sequences for LSTM
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the Model
    history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    # Predict Volatility
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return model, scaler, scaled_data, y_test, y_pred, train_size, history

# Function to predict future volatility
def predict_future_volatility(model, scaler, scaled_data, seq_length, days_ahead):
    last_sequence = scaled_data[-seq_length:]
    future_predictions = []
    for i in range(days_ahead):
        X_pred = last_sequence.reshape((1, seq_length, 1))
        next_pred = model.predict(X_pred, verbose=0)
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred, axis=0)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 3. Train models for each ticker and calculate accuracy
seq_length = 60
future_days = 30
models = {}
predictions = {}
accuracy_metrics = {}

for ticker in tickers:
    print(f"Training model for {ticker}...")
    model, scaler, scaled_data, y_test, y_pred, train_size, history = build_and_train_model(data_dict[ticker]['Volatility'], seq_length)
    future_volatility = predict_future_volatility(model, scaler, scaled_data, seq_length, future_days)
    
    # Calculate accuracy metrics
    metrics = calculate_accuracy_metrics(y_test.flatten(), y_pred.flatten())
    accuracy_metrics[ticker] = metrics
    
    models[ticker] = {
        'model': model,
        'scaler': scaler,
        'scaled_data': scaled_data,
        'y_test': y_test,
        'y_pred': y_pred,
        'train_size': train_size,
        'future_volatility': future_volatility,
        'history': history
    }

# 4. Print accuracy reports
company_names = {'XOM': 'Exxon Mobil', 'CVX': 'Chevron', '^GSPC': 'S&P 500'}

for ticker in tickers:
    print_accuracy_report(accuracy_metrics[ticker], company_names[ticker])

# 5. Enhanced Visualization with Accuracy Metrics
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Oil Companies vs S&P 500 Volatility Analysis with Accuracy Metrics', fontsize=16, fontweight='bold')

colors = {'XOM': 'blue', 'CVX': 'red', '^GSPC': 'green'}

for idx, ticker in enumerate(tickers):
    # Get test period dates
    test_start_idx = models[ticker]['train_size'] + seq_length
    test_dates = data_dict[ticker].index[test_start_idx:]
    
    # Get corresponding stock prices for the test period
    test_prices = original_data_dict[ticker]['Close'].loc[test_dates]
    
    # Create future dates for prediction
    last_date = data_dict[ticker].index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days, freq='D')
    
    # Volatility subplot
    ax1 = axes[idx, 0]
    ax1.plot(test_dates, models[ticker]['y_test'].flatten(), label='Actual Volatility', 
             color=colors[ticker], linewidth=2)
    ax1.plot(test_dates, models[ticker]['y_pred'].flatten(), label='Predicted Volatility', 
             color=colors[ticker], linewidth=2, linestyle='--', alpha=0.7)
    ax1.plot(future_dates, models[ticker]['future_volatility'].flatten(), 
             label='Future Volatility Prediction', color='orange', linewidth=2, linestyle=':')
    
    ax1.set_ylabel('Volatility (Standard Deviation)', fontsize=10, fontweight='bold')
    ax1.set_title(f'{company_names[ticker]} Volatility Prediction', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    # Add accuracy metrics
    metrics = accuracy_metrics[ticker]
    acc_text = f'R²: {metrics["R2"]:.3f}\nMAPE: {metrics["MAPE"]:.1f}%\nDir. Acc: {metrics["Directional_Accuracy"]:.1f}%'
    ax1.text(0.02, 0.98, acc_text, transform=ax1.transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Stock price subplot
    ax2 = axes[idx, 1]
    ax2.plot(test_dates, test_prices, label=f'{ticker} Stock Price', 
             color=colors[ticker], linewidth=1.5)
    ax2.set_ylabel('Stock Price ($)', fontsize=10, fontweight='bold')
    ax2.set_title(f'{company_names[ticker]} Stock Price', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    # Add price statistics
    price_stats = f'Avg: ${np.mean(test_prices):.2f}\nMax: ${np.max(test_prices):.2f}\nMin: ${np.min(test_prices):.2f}'
    ax2.text(0.02, 0.98, price_stats, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Training loss subplot
    ax3 = axes[idx, 2]
    history = models[ticker]['history']
    ax3.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    ax3.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    ax3.set_ylabel('Loss (MSE)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax3.set_title(f'{company_names[ticker]} Training History', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis dates for first two columns
    if idx == 2:  # Only for bottom plots
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])

plt.tight_layout()
plt.show()

# 6. Accuracy Comparison Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Accuracy Comparison Across Tickers', fontsize=16, fontweight='bold')

# Prepare data for comparison
metrics_comparison = pd.DataFrame(accuracy_metrics).T

# R² comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(range(len(tickers)), metrics_comparison['R2'], color=[colors[t] for t in tickers])
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('R² Score Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(tickers)))
ax1.set_xticklabels([company_names[t] for t in tickers])
ax1.grid(True, alpha=0.3)
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# MAPE comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(range(len(tickers)), metrics_comparison['MAPE'], color=[colors[t] for t in tickers])
ax2.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax2.set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(tickers)))
ax2.set_xticklabels([company_names[t] for t in tickers])
ax2.grid(True, alpha=0.3)
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# Directional Accuracy comparison
ax3 = axes[1, 0]
bars3 = ax3.bar(range(len(tickers)), metrics_comparison['Directional_Accuracy'], color=[colors[t] for t in tickers])
ax3.set_ylabel('Directional Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Directional Accuracy Comparison', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(tickers)))
ax3.set_xticklabels([company_names[t] for t in tickers])
ax3.grid(True, alpha=0.3)
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# RMSE comparison
ax4 = axes[1, 1]
bars4 = ax4.bar(range(len(tickers)), metrics_comparison['RMSE'], color=[colors[t] for t in tickers])
ax4.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax4.set_title('Root Mean Square Error', fontsize=12, fontweight='bold')
ax4.set_xticks(range(len(tickers)))
ax4.set_xticklabels([company_names[t] for t in tickers])
ax4.grid(True, alpha=0.3)
for i, bar in enumerate(bars4):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# 7. Overall Summary with Rankings
print("\n=== OVERALL MODEL PERFORMANCE SUMMARY ===")
print("\nModel Rankings (based on R² score):")
r2_rankings = metrics_comparison['R2'].sort_values(ascending=False)
for i, (ticker, r2) in enumerate(r2_rankings.items(), 1):
    print(f"{i}. {company_names[ticker]} (R² = {r2:.4f})")

print("\nModel Rankings (based on MAPE - lower is better):")
mape_rankings = metrics_comparison['MAPE'].sort_values(ascending=True)
for i, (ticker, mape) in enumerate(mape_rankings.items(), 1):
    print(f"{i}. {company_names[ticker]} (MAPE = {mape:.2f}%)")

print("\nModel Rankings (based on Directional Accuracy):")
dir_rankings = metrics_comparison['Directional_Accuracy'].sort_values(ascending=False)
for i, (ticker, dir_acc) in enumerate(dir_rankings.items(), 1):
    print(f"{i}. {company_names[ticker]} (Dir. Acc = {dir_acc:.1f}%)")

# 8. Volatility Comparison Plot with Accuracy
plt.figure(figsize=(14, 8))

# Get common date range for comparison
common_dates = None
for ticker in tickers:
    test_start_idx = models[ticker]['train_size'] + seq_length
    test_dates = data_dict[ticker].index[test_start_idx:]
    if common_dates is None:
        common_dates = test_dates
    else:
        common_dates = common_dates.intersection(test_dates)

# Plot volatility comparisons
for ticker in tickers:
    test_start_idx = models[ticker]['train_size'] + seq_length
    test_dates = data_dict[ticker].index[test_start_idx:]
    
    # Align dates
    aligned_volatility = pd.Series(models[ticker]['y_test'].flatten(), index=test_dates)
    aligned_volatility = aligned_volatility.loc[common_dates]
    
    r2_score = accuracy_metrics[ticker]['R2']
    plt.plot(common_dates, aligned_volatility, 
             label=f'{company_names[ticker]} (R²={r2_score:.3f})', 
             color=colors[ticker], linewidth=2)

plt.ylabel('Volatility (Standard Deviation)', fontsize=12, fontweight='bold')
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.title('Volatility Comparison with Model Accuracy: Oil Companies vs S&P 500', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 9. Final Analysis Summary with Accuracy
print("\n=== COMPREHENSIVE ANALYSIS SUMMARY ===")
for ticker in tickers:
    future_vol = models[ticker]['future_volatility']
    metrics = accuracy_metrics[ticker]
    
    print(f"\n{company_names[ticker]} ({ticker}):")
    print(f"  Model Performance:")
    print(f"    R² Score: {metrics['R2']:.4f}")
    print(f"    MAPE: {metrics['MAPE']:.2f}%")
    print(f"    Directional Accuracy: {metrics['Directional_Accuracy']:.1f}%")
    print(f"  Predicted volatility for the next {future_days} days:")
    print(f"    Average: {np.mean(future_vol):.2%}")
    print(f"    Maximum: {np.max(future_vol):.2%}")
    print(f"    Minimum: {np.min(future_vol):.2%}")
    
    # Calculate correlation between price changes and volatility
    test_start_idx = models[ticker]['train_size'] + seq_length
    test_dates = data_dict[ticker].index[test_start_idx:]
    test_prices = original_data_dict[ticker]['Close'].loc[test_dates]
    
    price_changes = test_prices.pct_change().dropna()
    volatility_aligned = pd.Series(models[ticker]['y_test'].flatten(), index=test_dates)
    volatility_aligned = volatility_aligned.loc[price_changes.index]
    
    correlation = np.corrcoef(price_changes, volatility_aligned)[0, 1]
    print(f"    Correlation between price changes and volatility: {correlation:.3f}")

print(f"\n=== KEY INSIGHTS ===")
best_r2_ticker = r2_rankings.index[0]
best_mape_ticker = mape_rankings.index[0]
best_dir_ticker = dir_rankings.index[0]

print(f"• Best overall model (R²): {company_names[best_r2_ticker]} with R² = {r2_rankings.iloc[0]:.4f}")
print(f"• Most accurate predictions (MAPE): {company_names[best_mape_ticker]} with MAPE = {mape_rankings.iloc[0]:.2f}%")
print(f"• Best directional prediction: {company_names[best_dir_ticker]} with {dir_rankings.iloc[0]:.1f}% accuracy")

# Average model performance
avg_r2 = metrics_comparison['R2'].mean()
avg_mape = metrics_comparison['MAPE'].mean()
avg_dir_acc = metrics_comparison['Directional_Accuracy'].mean()

print(f"• Average model performance: R² = {avg_r2:.4f}, MAPE = {avg_mape:.2f}%, Dir. Acc = {avg_dir_acc:.1f}%")