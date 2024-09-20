from flask import Flask, render_template
import requests
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

@app.route('/')
def index():
    coin_id = 'bitcoin'
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    parameters = {
        'vs_currency': 'nzd',
        'days': '365',
        'interval': 'daily'
    }

    response = requests.get(url, params=parameters)

    if response.status_code == 200:
        data = response.json()
        prices = data['prices']

        # DataFrame for Prophet
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['date', 'price']]
        df.rename(columns={'date': 'ds', 'price': 'y'}, inplace=True)

        # Forecast
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        last_7_days = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(8)

        last_7_days.rename(columns={'ds': 'Date', 'yhat': 'Price', 'yhat_lower':'Lower', 'yhat_upper':'Upper'}, inplace=True)

        # Convert date to string and round the price
        last_7_days['Date'] = last_7_days['Date'].dt.strftime('%Y-%m-%d')
        last_7_days['Price'] = last_7_days['Price'].round().astype(int)
        last_7_days['Lower'] = last_7_days['Lower'].round().astype(int)
        last_7_days['Upper'] = last_7_days['Upper'].round().astype(int)

        # Reorder the columns explicitly
        last_7_days = last_7_days[['Date', 'Price', 'Lower', 'Upper']]

        forecast_data = last_7_days.to_dict(orient='records')  # Convert to list of dictionaries
        return render_template('index.html', forecast=forecast_data)

    else:
        return "Error fetching data"

if __name__ == "__main__":
    app.run(debug=True)
