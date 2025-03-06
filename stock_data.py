import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import io
import base64
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import traceback

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Also fetch historical data for the chart
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        hist_data = stock.history(start=start_date, end=end_date)
        
        # Generate historical price plot
        if not hist_data.empty:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(hist_data['Close'])
            plt.title(f'{symbol} Stock Price - Past Year')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save figure to a buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            chart = base64.b64encode(buf.getvalue()).decode('utf-8')
        else:
            chart = None
        
        data = {
            "Current Price": info.get("currentPrice", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "Price-to-Book Ratio": info.get("priceToBook", "N/A"),
            "Debt-to-Equity Ratio": info.get("debtToEquity", "N/A"),
            "Return on Equity (ROE)": info.get("returnOnEquity", "N/A"),
            "Revenue Growth": info.get("revenueGrowth", "N/A"),
            "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Day High": info.get("dayHigh", "N/A"),
            "Day Low": info.get("dayLow", "N/A"),
            "chart": chart
        }
        return data
    except Exception as e:
        return {"Error": str(e)}

def predict_stock_prices(ticker):
    print(f"\n--- Starting stock price prediction for {ticker} ---")
    
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Calculate the start date (exactly 3 years before today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)  # Exactly 3 years
        
        print(f"Fetching data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch data using yfinance
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if data.empty:
            print(f"No data available for ticker: {ticker}")
            return f"No data available for ticker: {ticker}", None
        
        print(f"Successfully fetched {len(data)} data points")
        
        # Use only the 'Close' prices for prediction
        df1 = data['Close']
        
        # Generate historical price plot
        print("Generating historical price plot...")
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(df1)
        plt.title(f'{ticker} Stock Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xticks(
            ticks=df1.index[::len(df1)//12],
            labels=pd.to_datetime(df1.index[::len(df1)//12]).strftime("%b '%y"),
            rotation=45
        )
        plt.tight_layout()
        
        # Save figure to a buffer
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png')
        plt.close(fig1)
        buf1.seek(0)
        historical_plot = base64.b64encode(buf1.getvalue()).decode('utf-8')
        
        print("Normalizing data...")
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        df1_values = scaler.fit_transform(np.array(df1).reshape(-1, 1))

        # Split the data into training and test sets
        training_size = int(len(df1_values) * 0.65)
        test_size = len(df1_values) - training_size
        train_data, test_data = df1_values[0:training_size, :], df1_values[training_size:len(df1_values), :1]

        # Create dataset matrix
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        # Prepare the datasets
        time_step = 50
        print("Creating training and test datasets...")
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
        
        if len(X_train) == 0 or len(X_test) == 0:
            print("Not enough data points for prediction after creating sequences")
            return "Not enough historical data for prediction. Try a stock with more history.", None

        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2)  # [batch, seq_len, features]
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(2)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Initialize model
        print("Initializing LSTM model...")
        input_dim = 1
        hidden_dim = 50
        model = LSTMModel(input_dim, hidden_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        print("Training model...")
        num_epochs = 100
        best_val_loss = float('inf')
        patience = 40
        counter = 0
        best_model = None
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    y_pred = model(X_batch)
                    val_loss += criterion(y_pred.squeeze(), y_batch).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                counter = 0
            else:
                counter += 1
                
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        # Load best model
        if best_model:
            model.load_state_dict(best_model)
        
        print("Making predictions...")
        # Make predictions
        model.eval()
        with torch.no_grad():
            train_predict = model(X_train_tensor).numpy()
            test_predict = model(X_test_tensor).numpy()
            
        # Transform back to original scale
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
        
        # Calculate RMSE
        print("Calculating metrics...")
        # Fix for the "inconsistent samples" error:
        # Reshape the arrays to ensure they have compatible dimensions
        y_train_reshaped = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_reshaped = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        train_rmse = math.sqrt(mean_squared_error(y_train_reshaped, train_predict))
        test_rmse = math.sqrt(mean_squared_error(y_test_reshaped, test_predict))
        
        # Prepare prediction plot
        look_back = time_step
        trainPredictPlot = np.empty_like(df1_values)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        
        testPredictPlot = np.empty_like(df1_values)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1_values)-1, :] = test_predict
        
        # Plot the predictions
        print("Generating prediction plot...")
        fig2 = plt.figure(figsize=(10, 5))
        plt.plot(scaler.inverse_transform(df1_values), label='Actual Prices')
        plt.plot(trainPredictPlot, label='Train Predictions')
        plt.plot(testPredictPlot, label='Test Predictions')
        plt.title(f'{ticker} Stock Price Prediction using LSTM')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(
            ticks=range(0, len(df1_values), len(df1_values)//12),
            labels=pd.to_datetime(data.index[::len(df1_values)//12]).strftime("%b '%y"),
            rotation=45
        )
        plt.legend()
        plt.tight_layout()
        
        # Save figure to buffer
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        plt.close(fig2)
        buf2.seek(0)
        prediction_plot = base64.b64encode(buf2.getvalue()).decode('utf-8')
        
        # Generate future forecast (enhanced with more information)
        print("Generating forecast...")
        future_steps = 30
        forecast_input = torch.FloatTensor(test_data[-time_step:]).view(1, time_step, 1)
        forecast = []
        
        model.eval()
        with torch.no_grad():
            for _ in range(future_steps):
                next_pred = model(forecast_input).item()
                forecast.append(next_pred)
                next_item = torch.FloatTensor([[next_pred]]).view(1, 1, 1)
                forecast_input = torch.cat((forecast_input[:, 1:, :], next_item), dim=1)
        
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        
        # Plot the forecasted prices with improved visuals
        print("Generating forecast plot...")
        fig3 = plt.figure(figsize=(12, 6))
        
        # Get the dates for the forecast
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
        
        # Plot the last 60 days of actual data for context
        context_days = 60
        if len(data) > context_days:
            context_data = data['Close'][-context_days:]
            plt.plot(context_data.index, context_data, label='Historical Prices', color='blue')
        else:
            plt.plot(data.index, data['Close'], label='Historical Prices', color='blue')
        
        # Plot the forecast - keep only the red line without price annotations
        plt.plot(future_dates, forecast, label='30-Day Forecast', color='red', linewidth=2)
        
        # Add vertical line to mark the separation between historical and forecast
        plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
        
        # Calculate the percentage change but don't display exact prices
        current_price = float(data['Close'].iloc[-1])
        forecast_end_price = forecast[-1][0]
        pct_change = ((forecast_end_price - current_price) / current_price) * 100
        
        # Add a simple trend indicator without exact prices
        plt.text(future_dates[-1], forecast[-1][0], f' ({pct_change:+.2f}%)', 
                 verticalalignment='bottom', color='darkred')
        
        plt.title(f'{ticker} Stock Price 30-Day Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')  # Removed $ symbol
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure to buffer
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format='png')
        plt.close(fig3)
        buf3.seek(0)
        forecast_plot = base64.b64encode(buf3.getvalue()).decode('utf-8')
        
        metrics = f"Model trained on 3 years of data. Training RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}"
        print("Prediction complete successfully")
        
        # Only return the forecast plot
        return metrics, forecast_plot
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        return f"Error during prediction: {str(e)}", None
