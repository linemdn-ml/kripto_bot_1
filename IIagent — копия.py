import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import os
import sys
import logging
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv
from pathlib import Path

# ---------- ДОПОЛНИТЕЛЬНЫЕ ИМПОРТЫ ДЛЯ ИИ ----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------- ЗАГРУЗКА .env ----------
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET = os.getenv("BYBIT_SECRET")

if not DISCORD_WEBHOOK_URL:
    raise EnvironmentError("Не найден DISCORD_WEBHOOK_URL")

# ---------- ПЕРЕКЛЮЧАТЕЛИ ----------
AUTO_TRADE = False              # Сделки открываются автоматически (демо‑счёт)
USE_ML = True                  # Включаем ИИ-модель
ML_CONFIDENCE = 0.75           # Порог уверенности 75%

# ---------- НАСТРОЙКИ СТРАТЕГИИ ----------
SYMBOLS = [
    'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'LINK/USDT:USDT', 'LTC/USDT:USDT', 'BCH/USDT:USDT',
    'DOT/USDT:USDT', 'POL/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'FIL/USDT:USDT',
    'AAVE/USDT:USDT', 'AVAX/USDT:USDT', 'NEAR/USDT:USDT', 'APT/USDT:USDT', 'ARB/USDT:USDT',
    'OP/USDT:USDT', 'SUI/USDT:USDT', 'SEI/USDT:USDT', 'INJ/USDT:USDT', 'TIA/USDT:USDT',
    'RENDER/USDT:USDT', 'WLD/USDT:USDT', 'TAO/USDT:USDT', 'WIF/USDT:USDT',
    'ENA/USDT:USDT', 'EIGEN/USDT:USDT', 'STRK/USDT:USDT',
    'LDO/USDT:USDT', 'PENDLE/USDT:USDT', 'JTO/USDT:USDT', 'DYM/USDT:USDT', 'ZRO/USDT:USDT',
    'ONDO/USDT:USDT', 'HBAR/USDT:USDT', 'GRT/USDT:USDT', 'S/USDT:USDT',
    'MANA/USDT:USDT', 'SAND/USDT:USDT', 'AXS/USDT:USDT',
    'THETA/USDT:USDT', '1000PEPE/USDT:USDT', '1000BONK/USDT:USDT',
    '1000FLOKI/USDT:USDT', 'PEOPLE/USDT:USDT', 'TRX/USDT:USDT', 'STX/USDT:USDT'
]

TIMEFRAME = '15m'               # 15-минутные свечи

# --- Nadaraya-Watson Envelope ---
NW_BANDWIDTH = 8
NW_MULTIPLIER = 2.5
NW_LOOKBACK = 50

# --- RSI ---
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

SUPPRESS_SECONDS = 1 * 60

# --- Риск-менеджмент ---
ORDER_AMOUNT_USDT = 100
STOP_LOSS_ATR_MULT = 0.9
TAKE_PROFIT_ATR_MULT = 1.3
MAX_OPEN_POSITIONS = 5
ATR_PERIOD = 14

# --- Параметры ML-модели ---
ML_LOOKBACK = 500
ML_RETRAIN_INTERVAL = 3600
ML_TARGET_FORWARD = 3          # прогноз на 3 свечи вперёд (45 минут)

# ---------- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ML ----------
ml_model = None
ml_scaler = StandardScaler()
ml_trained = False
ml_last_train_time = 0
ml_lock = threading.Lock()

# ---------- ЛОГГИРОВАНИЕ ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bot_nw_rsi_ml.log"), logging.StreamHandler()]
)

# ========== БИРЖИ ==========
exchange_public = ccxt.bybit({
    'enableRateLimit': True,
    'timeout': 30000,
    'options': {'defaultType': 'linear', 'fetchMarkets': ['linear']},
    'urls': {'api': {'public': 'https://api.bybit.com'}},
})
exchange_public.load_markets()

exchange_private = ccxt.bybit({
    'apiKey': BYBIT_API_KEY,
    'secret': BYBIT_SECRET,
    'enableRateLimit': True,
    'timeout': 30000,
    'options': {'defaultType': 'linear', 'recvWindow': 10000},
    'urls': {'api': {'private': 'https://api-demo.bybit.com'}, 'www': 'https://www.bybit.com'},
})

try:
    time_diff = exchange_private.load_time_difference()
    logging.info(f"Синхронизация времени с Bybit выполнена, разница: {time_diff} мс")
except Exception as e:
    logging.warning(f"Не удалось синхронизировать время с Bybit: {e}")

exchange_private.markets = exchange_public.markets

# ---------- ОТПРАВКА В DISCORD ----------
def send_discord_embed(title: str, description: str, color: int, fields: list = None):
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    if fields:
        embed["fields"] = fields
    payload = {"embeds": [embed]}
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code == 204:
            logging.info("Сообщение отправлено в Discord")
        else:
            logging.error(f"Ошибка Discord: {resp.status_code} {resp.text}")
    except Exception as e:
        logging.error(f"Ошибка отправки Discord: {e}")

def send_signal_discord(signal_type, symbol, price, proba):
    emoji = "🟢" if signal_type == 'LONG' else "🔴"
    title = f"{emoji} {signal_type} {symbol} (ML)"
    description = f"Цена: {price:.4f}\nУверенность модели: {proba:.2%}"
    send_discord_embed(title, description, 0x00ff00 if signal_type == 'LONG' else 0xff0000)

# ---------- ПРОВЕРКА ПОЗИЦИЙ ----------
def count_open_positions() -> int:
    try:
        positions = exchange_private.fetch_positions()
        active = [p for p in positions if p['contracts'] > 0]
        return len(active)
    except Exception as e:
        logging.error(f"Ошибка получения позиций: {e}")
        return 0

# ---------- ОТКРЫТИЕ ПОЗИЦИИ ----------
def open_bybit_demo_position(symbol: str, side: str, atr: float) -> bool:
    if not AUTO_TRADE:
        logging.info(f"Автотрейдинг отключён. Сигнал {symbol} не привёл к сделке.")
        return False
    if count_open_positions() >= MAX_OPEN_POSITIONS:
        logging.info(f"Лимит позиций достигнут. Пропускаю {symbol}")
        return False
    try:
        ticker = exchange_public.fetch_ticker(symbol)
        current_price = ticker['last']
        if side == 'LONG':
            sl_price = current_price - STOP_LOSS_ATR_MULT * atr
            tp_price = current_price + TAKE_PROFIT_ATR_MULT * atr
        else:
            sl_price = current_price + STOP_LOSS_ATR_MULT * atr
            tp_price = current_price - TAKE_PROFIT_ATR_MULT * atr
        sl_price = round(sl_price, 4)
        tp_price = round(tp_price, 4)
        amount = ORDER_AMOUNT_USDT / current_price
        order = exchange_private.create_order(
            symbol=symbol,
            type='market',
            side='buy' if side == 'LONG' else 'sell',
            amount=amount,
            params={'stopLoss': sl_price, 'takeProfit': tp_price}
        )
        logging.info(f"ДЕМО {side} {symbol} | Цена: {current_price} | SL: {sl_price} | TP: {tp_price}")
        return True
    except Exception as e:
        logging.error(f"Ошибка открытия позиции {symbol}: {e}")
        return False

# ---------- NADARAYA-WATSON ENVELOPE ----------
def nadaraya_watson_non_repaint(close_prices: np.ndarray, bandwidth: float, multiplier: float):
    n = len(close_prices)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan
    regression = np.full(n, np.nan)
    variance = np.full(n, np.nan)
    for i in range(n):
        y_local = close_prices[:i+1]
        x_local = np.arange(i+1)
        if len(y_local) < 2:
            continue
        weights = np.exp(-0.5 * ((x_local - i) / bandwidth) ** 2)
        sum_w = np.sum(weights)
        if sum_w == 0:
            regression[i] = y_local[-1]
        else:
            regression[i] = np.sum(weights * y_local) / sum_w
        if i > 0:
            sq_err = (y_local - regression[i]) ** 2
            variance[i] = np.sum(weights * sq_err) / sum_w
    last_reg = regression[-1]
    last_var = variance[-1] if not np.isnan(variance[-1]) else 0
    std_res = np.sqrt(last_var)
    upper = last_reg + multiplier * std_res
    lower = last_reg - multiplier * std_res
    return last_reg, upper, lower, std_res

# ---------- ПОДГОТОВКА ПРИЗНАКОВ ДЛЯ ML (УЛУЧШЕННАЯ) ----------
def prepare_features(df):
    close = df['close']
    high = df['h']
    low = df['l']
    volume = df['volume']

    features = pd.DataFrame(index=df.index)

    # --- Базовые ценовые признаки ---
    features['returns_1'] = close.pct_change(1)
    features['returns_3'] = close.pct_change(3)
    features['high_low_ratio'] = (high - low) / close

    # --- RSI (текущий и лаги) ---
    rsi = ta.rsi(close, length=14)
    features['rsi'] = rsi
    features['rsi_lag1'] = rsi.shift(1)
    features['rsi_lag2'] = rsi.shift(2)

    # --- ATR (нормированный) ---
    atr = ta.atr(high, low, close, length=14)
    features['atr'] = atr / close

    # --- Положение относительно NW конверта ---
    closes = close.values
    reg, upper, lower, _ = nadaraya_watson_non_repaint(closes, NW_BANDWIDTH, NW_MULTIPLIER)
    features['dist_to_lower'] = (close.iloc[-1] - lower) / close.iloc[-1] if not np.isnan(lower) else 0
    features['dist_to_upper'] = (upper - close.iloc[-1]) / close.iloc[-1] if not np.isnan(upper) else 0

    # --- Объёмные признаки ---
    features['volume_ratio'] = volume / volume.rolling(20).mean()

    # --- MACD (гистограмма и лаги) ---
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    macd_hist = macd[f'MACDh_12_26_9']
    features['macd_hist'] = macd_hist
    features['macd_hist_lag1'] = macd_hist.shift(1)
    features['macd_hist_lag2'] = macd_hist.shift(2)

    # --- Stochastic RSI (K и лаги) ---
    stoch = ta.stochrsi(close, length=14, smoothK=3, smoothD=3)
    k_col = [c for c in stoch.columns if 'STOCHRSIk' in c][0]
    stoch_k = stoch[k_col]
    features['stoch_k'] = stoch_k
    features['stoch_k_lag1'] = stoch_k.shift(1)
    features['stoch_k_lag2'] = stoch_k.shift(2)

    # --- Трендовый признак (положение относительно EMA 50) ---
    ema50 = ta.ema(close, length=50)
    features['ema_50_ratio'] = ema50 / close

    # --- Корреляция с BTC/USDT (если доступен) ---
    try:
        btc_bars = exchange_public.fetch_ohlcv('BTC/USDT:USDT', timeframe=TIMEFRAME, limit=2)
        if btc_bars and len(btc_bars) >= 2:
            btc_return = (btc_bars[-1][4] - btc_bars[-2][4]) / btc_bars[-2][4]
            features['btc_return'] = btc_return
        else:
            features['btc_return'] = 0.0
    except:
        features['btc_return'] = 0.0

    # Убираем бесконечности и NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    return features

# ---------- ОБУЧЕНИЕ ML-МОДЕЛИ ----------
def train_ml_model(symbol):
    global ml_model, ml_scaler, ml_trained, ml_last_train_time
    try:
        logging.info(f"Обучение ML-модели для {symbol}...")
        bars = exchange_public.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=ML_LOOKBACK + ML_TARGET_FORWARD + 20)
        if len(bars) < 100:
            logging.warning(f"Недостаточно данных для обучения {symbol}")
            return

        df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'close', 'v'])
        df.rename(columns={'v': 'volume'}, inplace=True)
        df.set_index('t', inplace=True)

        X = prepare_features(df)

        # Целевая переменная: рост через N свечей
        future_close = df['close'].shift(-ML_TARGET_FORWARD)
        y = (future_close > df['close']).astype(int)
        y = y.iloc[:len(X)]

        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if len(y) < 50:
            logging.warning("Слишком мало размеченных данных")
            return

        X_scaled = ml_scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_scaled, y)
        score = model.score(X_scaled, y)
        logging.info(f"Модель обучена. Точность на обучающих данных: {score:.2%}")

        with ml_lock:
            ml_model = model
            ml_scaler = StandardScaler().fit(X)
            ml_trained = True
            ml_last_train_time = time.time()

    except Exception as e:
        logging.error(f"Ошибка при обучении ML-модели: {e}")

# ---------- ПОЛУЧЕНИЕ СИГНАЛА ОТ ML ----------
def ml_predict_signal(features_df):
    global ml_model, ml_scaler, ml_trained
    if not ml_trained or ml_model is None:
        return None, 0.0
    try:
        X = features_df.iloc[-1:].copy()
        X_scaled = ml_scaler.transform(X)
        proba = ml_model.predict_proba(X_scaled)[0]
        prob_up = proba[1] if len(proba) > 1 else 0.5
        if prob_up > ML_CONFIDENCE:
            return 'LONG', prob_up
        elif (1 - prob_up) > ML_CONFIDENCE:
            return 'SHORT', 1 - prob_up
        else:
            return None, prob_up
    except Exception as e:
        logging.error(f"Ошибка предсказания ML: {e}")
        return None, 0.0

# ---------- РАСЧЁТ СИГНАЛА (основной цикл) ----------
last_signal_time = {}

def get_signal(symbol: str) -> dict | None:
    try:
        time.sleep(0.3)

        now = time.time()
        if USE_ML and (now - ml_last_train_time) > ML_RETRAIN_INTERVAL:
            train_ml_model(symbol)

        for attempt in range(3):
            try:
                bars = exchange_public.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=NW_LOOKBACK + 10)
                if bars and len(bars) > 0:
                    bars = bars[:-1]
                break
            except Exception as e:
                err_str = str(e)
                if '10006' in err_str or 'Rate Limit' in err_str or 'Too many visits' in err_str:
                    logging.warning(f"{symbol}: rate limit, попытка {attempt+1}/3, жду 5 сек...")
                    time.sleep(5)
                else:
                    raise e
        else:
            logging.error(f"{symbol}: не удалось получить свечи после 3 попыток")
            return None

        if len(bars) < 30:
            return None

        df = pd.DataFrame(bars, columns=['t', 'o', 'h', 'l', 'close', 'v'])
        df.rename(columns={'v': 'volume'}, inplace=True)

        close_prices = df['close'].values

        # RSI и ATR
        df['RSI'] = ta.rsi(df['close'], length=RSI_PERIOD)
        df['atr'] = ta.atr(df['h'], df['l'], df['close'], length=ATR_PERIOD)

        # Non-repainting Nadaraya-Watson
        regression, upper, lower, _ = nadaraya_watson_non_repaint(
            close_prices, bandwidth=NW_BANDWIDTH, multiplier=NW_MULTIPLIER
        )

        last_close = df['close'].iloc[-1]
        last_rsi = df['RSI'].iloc[-1]
        last_atr = df['atr'].iloc[-1]

        if np.isnan(regression) or np.isnan(last_rsi):
            return None

        # Если ИИ активен, используем его предсказание
        if USE_ML and ml_trained:
            features = prepare_features(df)
            ml_signal, confidence = ml_predict_signal(features)
            if ml_signal:
                if symbol in last_signal_time and (now - last_signal_time[symbol]) < SUPPRESS_SECONDS:
                    return None
                last_signal_time[symbol] = now
                return {
                    'type': ml_signal,
                    'price': last_close,
                    'rsi': last_rsi,
                    'lower': lower,
                    'upper': upper,
                    'regression': regression,
                    'atr': last_atr,
                    'confidence': confidence
                }
            else:
                return None

        # Fallback на правила
        if last_close <= lower and last_rsi < RSI_OVERSOLD:
            if symbol in last_signal_time and (now - last_signal_time[symbol]) < SUPPRESS_SECONDS:
                return None
            last_signal_time[symbol] = now
            return {
                'type': 'LONG',
                'price': last_close,
                'rsi': last_rsi,
                'lower': lower,
                'upper': upper,
                'regression': regression,
                'atr': last_atr
            }

        if last_close >= upper and last_rsi > RSI_OVERBOUGHT:
            if symbol in last_signal_time and (now - last_signal_time[symbol]) < SUPPRESS_SECONDS:
                return None
            last_signal_time[symbol] = now
            return {
                'type': 'SHORT',
                'price': last_close,
                'rsi': last_rsi,
                'lower': lower,
                'upper': upper,
                'regression': regression,
                'atr': last_atr
            }

    except Exception as e:
        logging.error(f"Ошибка расчёта {symbol}: {e}")
    return None

# ---------- ОЖИДАНИЕ 15-МИНУТНОЙ СВЕЧИ ----------
def wait_for_next_candle():
    now = datetime.now(timezone.utc)
    minutes_past = now.minute % 15
    seconds_past = now.second + now.microsecond / 1_000_000
    wait_seconds = (15 - minutes_past) * 60 - seconds_past
    if wait_seconds < 0:
        wait_seconds += 15 * 60
    wait_seconds += 2
    logging.info(f"Ожидание {wait_seconds:.0f} сек. до следующей 15m свечи...")
    time.sleep(max(wait_seconds, 0))

# ---------- ОСНОВНОЙ ЦИКЛ ----------
def main():
    logging.info("Бот (Nadaraya-Watson Envelope + RSI) запущен с ИИ-модулем.")
    logging.info(f"Мониторинг {len(SYMBOLS)} пар, TF={TIMEFRAME}")
    logging.info(f"ИИ: {'активен' if USE_ML else 'отключён'}")
    logging.info(f"Порог уверенности ML: {ML_CONFIDENCE*100:.0f}%")
    logging.info(f"Автотрейдинг {'ВКЛЮЧЁН' if AUTO_TRADE else 'ОТКЛЮЧЁН'}.")

    send_discord_embed(
        "🤖 Бот запущен (NW + RSI + ML)",
        f"ИИ-модель RandomForest, уверенность > {ML_CONFIDENCE*100:.0f}%. "
        f"Правила NW+RSI работают как fallback. TF {TIMEFRAME}",
        0x9b59b6
    )

    if USE_ML:
        train_ml_model(SYMBOLS[0])

    while True:
        wait_for_next_candle()
        logging.info("Свеча закрыта. Начало анализа...")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_symbol = {executor.submit(get_signal, s): s for s in SYMBOLS}
            for future in as_completed(future_to_symbol):
                signal = future.result()
                if signal:
                    sym = future_to_symbol[future]
                    conf_str = f" (ML conf: {signal.get('confidence', 0):.2%})" if 'confidence' in signal else ""
                    text = (f"[{signal['type']}] {sym}{conf_str}\n"
                            f"Цена: {signal['price']:.4f}\n"
                            f"RSI: {signal['rsi']:.2f}\n"
                            f"Нижняя граница: {signal['lower']:.4f}\n"
                            f"Верхняя граница: {signal['upper']:.4f}\n"
                            f"Регрессия: {signal['regression']:.4f}\n"
                            f"ATR: {signal['atr']:.4f}")
                    print("\n" + "="*50)
                    print(text)
                    print("="*50 + "\n")

                    if 'confidence' in signal:
                        send_signal_discord(signal['type'], sym.split(':')[0], signal['price'], signal['confidence'])
                    else:
                        send_signal_discord(signal['type'], sym.split(':')[0], signal['price'], signal['rsi'])

                    open_bybit_demo_position(sym, signal['type'], signal['atr'])

                time.sleep(0.2)

if __name__ == "__main__":
    main()