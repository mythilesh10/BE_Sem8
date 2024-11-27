from flask import Flask, render_template, request, jsonify
import yfinance as yf
import talib
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
# from flask_talisman import Talisman
from pymongo import MongoClient
import requests

# Configure Flask app
app = Flask(__name__)
log_file = 'flask_app.log'
handler = RotatingFileHandler(log_file, maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Configure Flask-Talisman for CSP
# csp = {
#     'default-src': [
#         '\'self\''
#     ],
#     'style-src': [
#         '\'self\'',
#         'https://fonts.googleapis.com'
#     ],
#     'font-src': [
#         '\'self\'',
#         'https://fonts.gstatic.com'
#     ],
# }
# Talisman(app, content_security_policy=csp)

# Database configuration
DATABASE_URI = 'mysql+pymysql://root:Neevesh%40123@localhost:3306/trading_app'
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# MongoDB database connection
client = MongoClient('mongodb://localhost:27017/')
db = client['news_database']
collection = db['articles']

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
API_TOKEN = "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"  # Replace with your Hugging Face API token
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Model for storing raw data
class StockData(Base):
    __tablename__ = 'stock_data'
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)

# Model for storing processed data
class ProcessedData(Base):
    __tablename__ = 'processed_data'
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, nullable=False)
    ema_short = Column(Float)
    ema_long = Column(Float)
    macd = Column(Float)
    signal = Column(Float)
    rsi = Column(Float)
    mfi = Column(Float)
    sar = Column(Float)
    signal_label = Column(Integer)

# Model for storing predictions
class Predictions(Base):
    __tablename__ = 'predictions'
    datetime = Column(DateTime, primary_key=True)
    predicted_signal = Column(Integer)

Base.metadata.create_all(engine)

# Parameters
start_date = "2024-05-06"
end_date = "2024-05-25"
short_period = 12
long_period = 26

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def download_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval='5m')
        data.reset_index(inplace=True)
        data['Datetime'] = data['Datetime'].astype(str)
        session = Session()
        stock_data_list = [
            StockData(
                datetime=row['Datetime'],
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                adj_close=row['Adj Close'],
                volume=row['Volume']
            )
            for index, row in data.iterrows()
        ]
        session.bulk_save_objects(stock_data_list)
        session.commit()
        session.close()
        return data
    except Exception as e:
        logging.error("Error downloading data: %s", e)
        return None

def generate_features_labels():
    session = Session()
    data = pd.read_sql(session.query(StockData).statement, session.bind)
    session.close()
    data.set_index('datetime', inplace=True)

    data['EMA_short'] = talib.EMA(data['close'], timeperiod=short_period)
    data['EMA_long'] = talib.EMA(data['close'], timeperiod=long_period)
    data['macd'], data['signal'], _ = talib.MACD(data['close'])
    data['RSI'] = talib.RSI(data['close'])
    data['MFI'] = talib.MFI(data['high'], data['low'], data['close'], data['volume'])
    data['SAR'] = talib.SAR(data['high'], data['low'])

    for indicator in ['EMA_short', 'EMA_long', 'macd', 'signal', 'RSI', 'MFI', 'SAR']:
        data[indicator + '_prev'] = data[indicator].shift(1)
    data['close_prev'] = data['close'].shift(1)

    buy_signals = (
        ((data['EMA_short'] >= data['EMA_long']) & (data['EMA_short_prev'] < data['EMA_long_prev'])) |
        ((data['macd'] >= data['signal']) & (data['macd_prev'] < data['signal_prev'])) |
        ((data['RSI'] <= 30) & (data['RSI_prev'] > 30)) |
        ((data['MFI'] <= 20) & (data['MFI_prev'] > 20)) |
        ((data['SAR'] <= data['close']) & (data['SAR_prev'] > data['close_prev']))
    )

    sell_signals = (
        ((data['EMA_short'] <= data['EMA_long']) & (data['EMA_short_prev'] > data['EMA_long_prev'])) |
        ((data['macd'] <= data['signal']) & (data['macd_prev'] > data['signal_prev'])) |
        ((data['RSI'] >= 70) & (data['RSI_prev'] < 70)) |
        ((data['MFI'] >= 80) & (data['MFI_prev'] < 80)) |
        ((data['SAR'] >= data['close']) & (data['SAR_prev'] < data['close_prev']))
    )

    data['signal'] = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))
    data.dropna(inplace=True)

    session = Session()
    processed_data_list = [
        ProcessedData(
            datetime=index,
            ema_short=row['EMA_short'],
            ema_long=row['EMA_long'],
            macd=row['macd'],
            signal=row['signal'],
            rsi=row['RSI'],
            mfi=row['MFI'],
            sar=row['SAR'],
            signal_label=row['signal']
        )
        for index, row in data.iterrows()
    ]
    session.bulk_save_objects(processed_data_list)
    session.commit()
    session.close()

    X = data[['EMA_short', 'EMA_long', 'macd', 'signal', 'RSI', 'MFI', 'SAR']]
    y = data['signal']
    return X, y

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download_data', methods=['POST'])
def download_data_route():
    ticker = request.form['ticker']
    logging.info("Ticker received from frontend: %s", ticker)
    downloaded_data = download_data(ticker, start_date, end_date)
    if downloaded_data is not None:
        generate_features_labels()
        session = Session()
        processed_data = pd.read_sql(session.query(ProcessedData).statement, session.bind)
        session.close()
        features = processed_data[['ema_short', 'ema_long', 'macd', 'signal', 'rsi', 'mfi', 'sar']]
        labels = processed_data['signal_label']
        logging.info("Features: %s", features)
        logging.info("Labels: %s", labels)
        return jsonify({'success': True, 'features': features.to_dict(), 'labels': labels.tolist()})
    else:
        return jsonify({'success': False, 'message': 'Failed to download data'})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        session = Session()
        processed_data = pd.read_sql(session.query(ProcessedData).statement, session.bind)
        session.close()
        processed_data.set_index('datetime', inplace=True)

        X = processed_data[['ema_short', 'ema_long', 'macd', 'signal', 'rsi', 'mfi', 'sar']]
        y = processed_data['signal_label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        predictions = pd.DataFrame({
            'datetime': processed_data.index.tolist(),
            'predicted_signal': np.concatenate([y_pred_train, y_pred_test])
        })

        predictions['datetime'] = predictions['datetime'].astype(str)
        predictions.to_sql('predictions', con=engine, if_exists='replace', index=False)

        return jsonify({'success': True, 'message': 'Model trained and predictions stored in the database.'})
    except Exception as e:
        logging.error("Error training model: %s", e)
        return jsonify({'success': False, 'message': 'Failed to train model'})

@app.route('/visualizeData', methods=['GET'])
def visualizeData():
    try:
        session = Session()
        processed_data = pd.read_sql(session.query(ProcessedData).statement, session.bind)
        session.close()
        processed_data.set_index('datetime', inplace=True)
        return jsonify({'success': True, 'data': processed_data.to_dict()})
    except Exception as e:
        logging.error("Error visualizing data: %s", e)
        return jsonify({'success': False, 'message': 'Failed to visualize data'})
    
@app.route('/predictions', methods=['GET'])
def show_predictions():
    try:
        session = Session()
        logging.info("Session created successfully.")
        
        predictions_data = pd.read_sql(session.query(Predictions).statement, session.bind)
        logging.info("Data fetched successfully from database.")
        
        session.close()
        
        predictions_data['datetime'] = predictions_data['datetime'].astype(str)
        
        return jsonify({'success': True, 'data': predictions_data.to_dict(orient='records')})
    except Exception as e:
        logging.error("Error fetching predictions: %s", e)
        return jsonify({'success': False, 'message': f'Failed to fetch predictions: {str(e)}'})


@app.route('/news', methods=['GET'])
def news():
    articles = list(collection.find({}, {'_id': 0}))
    return render_template('news.html', articles=articles)

@app.route('/api/articles', methods=['GET'])
def get_articles():
    articles = list(collection.find({}, {'_id': 0}))
    return jsonify(articles)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json
    index = int(data['index'])
    article = collection.find_one({'index': index})
    title = article['title']
    body = article['article_body']

    payload = {"inputs": body}
    sentiment_response = query(payload)
    sentiment = sentiment_response[0]['label']

    return jsonify({'index': index, 'sentiment': sentiment})

@app.route('/sentiment', methods=['GET'])
def sentiment():
    return render_template('sentiment.html')

if __name__ == '__main__':
    app.run(debug=True)
