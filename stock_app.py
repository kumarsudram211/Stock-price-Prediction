import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import streamlit as st

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# LSTM Model Definition
# ----------------------------
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

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.title("📈 Reliance Stock Forecasting App")

    # Load your dataset
    df = pd.read_csv(r"C:\Users\SANDILYA SUNDRAM\Desktop\Revision\archive\summer intern\reliance_stock_data.csv")
    df.columns = df.columns.str.strip()  # Remove spaces
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date', 'Close'], inplace=True)
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')
    df['Close'].interpolate(method='linear', inplace=True)

    st.header("🧾 Data Preview")
    st.write(df.head())

    st.header("📊 Visualizations")
    st.line_chart(df['Close'])

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("ADF Stationarity Test")
    result = adfuller(df['Close'].dropna())
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")

    st.header("🔮 Forecasting Models")
    model_choice = st.selectbox("Choose Model:", ["ARIMA", "SARIMAX", "Prophet", "SVR", "Random Forest", "LSTM"])

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    if model_choice == "ARIMA":
        p = st.slider("p", 0, 5, 1)
        d = st.slider("d", 0, 2, 1)
        q = st.slider("q", 0, 5, 1)
        if st.button("Run ARIMA"):
            model = ARIMA(train['Close'], order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            st.line_chart(pd.DataFrame({"Actual": test['Close'], "Forecast": forecast}, index=test.index))

    elif model_choice == "SARIMAX":
        p = st.slider("p", 0, 5, 1)
        d = st.slider("d", 0, 2, 1)
        q = st.slider("q", 0, 5, 1)
        P = st.slider("P(seasonal)", 0, 5, 1)
        s = st.slider("Seasonal Period", 1, 30, 12)
        if st.button("Run SARIMAX"):
            model = SARIMAX(train['Close'], order=(p, d, q), seasonal_order=(P, d, q, s))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            st.line_chart(pd.DataFrame({"Actual": test['Close'], "Forecast": forecast}, index=test.index))

    elif model_choice == "Prophet":
        if st.button("Run Prophet"):
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            m = Prophet()
            m.fit(prophet_df.iloc[:train_size])
            future = m.make_future_dataframe(periods=len(test))
            forecast = m.predict(future)
            fig = m.plot(forecast)
            st.pyplot(fig)

    elif model_choice == "SVR":
        if st.button("Run SVR"):
            model = SVR()
            model.fit(np.arange(train_size).reshape(-1, 1), train['Close'])
            preds = model.predict(np.arange(train_size, len(df)).reshape(-1, 1))
            st.line_chart(pd.DataFrame({"Actual": test['Close'], "Forecast": preds}, index=test.index))

    elif model_choice == "Random Forest":
        if st.button("Run Random Forest"):
            model = RandomForestRegressor()
            model.fit(np.arange(train_size).reshape(-1, 1), train['Close'])
            preds = model.predict(np.arange(train_size, len(df)).reshape(-1, 1))
            st.line_chart(pd.DataFrame({"Actual": test['Close'], "Forecast": preds}, index=test.index))

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
            train_seq, train_labels = create_sequences(train['Close'].values, seq_len)
            test_seq, _ = create_sequences(test['Close'].values, seq_len)

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

            preds = []
            with torch.no_grad():
                for seq in test_seq:
                    pred = model(seq.unsqueeze(0))
                    preds.append(pred.item())

            st.line_chart(pd.DataFrame({
                "Actual": test['Close'].values[seq_len:],
                "Forecast": preds
            }, index=test.index[seq_len:]))

if __name__ == "__main__":
    main()