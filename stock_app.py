import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.stattools import adfuller

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50)
        c0 = torch.zeros(2, x.size(0), 50)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def main():
    st.title("📈 Reliance Stock Forecasting App")

    df = pd.read_csv(r"C:\Users\SANDILYA SUNDRAM\Desktop\Revision\archive\summer intern\reliance_stock_data.csv")
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date', 'Close'], inplace=True)
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')
    df['Close'] = df['Close'].interpolate(method='linear')

    st.header("📊 Data Preview")
    st.write(df.tail())

    st.line_chart(df['Close'])

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("ADF Stationarity Test")
    result = adfuller(df['Close'].dropna())
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")

    st.header("🔮 Forecasting Models")
    model_choice = st.selectbox("Choose Model:", ["ARIMA", "SARIMAX", "Prophet", "SVR", "Random Forest", "LSTM"])
    forecast_days = st.slider("📅 Select number of days to forecast:", 1, 60, 30)

    if model_choice == "ARIMA":
        p = st.slider("p", 0, 5, 1)
        d = st.slider("d", 0, 2, 1)
        q = st.slider("q", 0, 5, 1)
        if st.button("Run ARIMA"):
            model = ARIMA(df['Close'], order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_days)
            future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            st.line_chart(pd.concat([df['Close'], pd.Series(forecast, index=future_index)]))

    elif model_choice == "SARIMAX":
        p = st.slider("p", 0, 5, 1)
        d = st.slider("d", 0, 2, 1)
        q = st.slider("q", 0, 5, 1)
        P = st.slider("P(seasonal)", 0, 5, 1)
        s = st.slider("Seasonal Period", 1, 30, 12)
        if st.button("Run SARIMAX"):
            model = SARIMAX(df['Close'], order=(p, d, q), seasonal_order=(P, d, q, s))
            result_model = model.fit()
            forecast = result_model.forecast(steps=forecast_days)
            future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            st.line_chart(pd.concat([df['Close'], pd.Series(forecast, index=future_index)]))

    elif model_choice == "Prophet":
        if st.button("Run Prophet"):
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=forecast_days)
            forecast = m.predict(future)
            fig = m.plot(forecast)
            st.pyplot(fig)

    elif model_choice == "SVR":
        if st.button("Run SVR"):
            model = SVR()
            x = np.arange(len(df)).reshape(-1, 1)
            model.fit(x, df['Close'])
            future_x = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
            preds = model.predict(future_x)
            future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            st.line_chart(pd.concat([df['Close'], pd.Series(preds, index=future_index)]))

    elif model_choice == "Random Forest":
        if st.button("Run Random Forest"):
            model = RandomForestRegressor()
            x = np.arange(len(df)).reshape(-1, 1)
            model.fit(x, df['Close'])
            future_x = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
            preds = model.predict(future_x)
            future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            st.line_chart(pd.concat([df['Close'], pd.Series(preds, index=future_index)]))

    elif model_choice == "LSTM":
        if st.button("Run LSTM"):
            def create_sequences(data, seq_len=10):
                xs, ys = [], []
                for i in range(len(data) - seq_len):
                    x = data[i:i + seq_len]
                    y = data[i + seq_len]
                    xs.append(x)
                    ys.append(y)
                return torch.tensor(xs).float().unsqueeze(-1), torch.tensor(ys).float()

            seq_len = 10
            series = df['Close'].values
            train_seq, train_labels = create_sequences(series)

            loader = DataLoader(TensorDataset(train_seq, train_labels), batch_size=16, shuffle=False)
            model = StockLSTM()
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(50):
                for x_batch, y_batch in loader:
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = loss_fn(output.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()

            last_seq = torch.tensor(series[-seq_len:]).float().unsqueeze(0).unsqueeze(-1)
            preds = []
            with torch.no_grad():
                for _ in range(forecast_days):
                    pred = model(last_seq)
                    preds.append(pred.item())
                    last_seq = torch.cat((last_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(2)), dim=1)

            future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            st.line_chart(pd.concat([df['Close'], pd.Series(preds, index=future_index)]))

if __name__ == "__main__":
    main()
