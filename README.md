Крипто-бот для Bybit: сканирует 45+ USDT-пар на 15m, генерирует LONG/SHORT сигналы на основе Nadaraya-Watson Envelope + RSI и ML-модели (RandomForest), отправляет уведомления в Discord. Опционально — автотрейдинг на демо-счёте.
2. Topics (теги для репозитория)
text
crypto  trading-bot  bybit  ccxt  python  machine-learning  random-forest  technical-analysis  rsi  nadaraya-watson  discord-webhook  futures  algorithmic-trading  quantitative-finance
3. README.md
markdown
# Bybit Signal Bot

Крипто-бот для мониторинга рынка на бирже **Bybit** (USDT-фьючерсы) и генерации торговых сигналов **LONG/SHORT** с отправкой уведомлений в **Discord**.

Работает на таймфрейме **15 минут** и совмещает два независимых источника сигналов: ML-модель (RandomForest) и классическую техническую логику (Nadaraya-Watson Envelope + RSI).

---

## Возможности

- 📊 **Мониторинг 45+ USDT-пар** одновременно (ADA, DOGE, LINK, AVAX, ARB, SUI, TAO, WIF, 1000PEPE и др.)
- ⏱ **Строго по закрытию 15m-свечи** — без репейнта и ложных срабатываний на незакрытых барах
- 🧠 **ML-модель (RandomForest)** — прогноз направления через 3 свечи вперёд с порогом уверенности 75%
- 📐 **Nadaraya-Watson Envelope (non-repainting)** + **RSI** — rule-based fallback для сигналов
- 💬 **Discord-уведомления** — rich-embed карточки с ценой, RSI, границами канала и ATR
- 🛡 **Риск-менеджмент** — автоматические SL/TP от ATR, лимит открытых позиций, подавление дубликатов сигналов
- 🤖 **Опциональный автотрейдинг** — открытие сделок на демо-счёте Bybit (по умолчанию выключен)

---

## Стек

`Python` · `ccxt` · `pandas` · `pandas_ta` · `scikit-learn` · `numpy` · `Discord Webhooks` · `Bybit Unified API`

---

## Установка

```bash
git clone https://github.com/username/bybit-signal-bot.git
cd bybit-signal-bot
pip install -r requirements.txt
Создайте файл .env в корне проекта:

env
DISCORD_WEBHOOK_URL=your_discord_webhook_url
BYBIT_API_KEY=your_bybit_api_key
BYBIT_SECRET=your_bybit_secret
Запуск:

bash
python bot.py
Настройки
Основные параметры находятся в начале bot.py:

Параметр	По умолчанию	Описание
TIMEFRAME	15m	Таймфрейм анализа
AUTO_TRADE	False	Автооткрытие сделок на демо Bybit
USE_ML	True	Использовать ML-модуль
ML_CONFIDENCE	0.75	Порог уверенности модели
NW_BANDWIDTH / NW_MULTIPLIER	8 / 2.5	Параметры NW-конверта
RSI_OVERSOLD / RSI_OVERBOUGHT	30 / 70	Уровни RSI
STOP_LOSS_ATR_MULT / TAKE_PROFIT_ATR_MULT	0.9 / 1.3	Множители ATR для SL/TP
MAX_OPEN_POSITIONS	5	Лимит одновременных позиций
ORDER_AMOUNT_USDT	100	Размер позиции в USDT
Как это работает
Бот ждёт закрытия 15-минутной свечи.

Для каждого символа загружает последние 60 свечей и рассчитывает индикаторы.

Проверяет сигнал от ML-модели (обучается автоматически и переобучается раз в час).

Если ML-сигнала нет — срабатывает rule-based логика:

LONG: цена пробила нижнюю границу NW-конверта + RSI < 30

SHORT: цена пробила верхнюю границу NW-конверта + RSI > 70

При наличии сигнала — отправляет уведомление в Discord и (если AUTO_TRADE = True) открывает позицию на демо-счёте с рассчитанными SL и TP.

Дисклеймер
Проект создан в образовательных целях. Торговля криптовалютными фьючерсами сопряжена с высоким риском потери депозита. Автор не несёт ответственности за финансовые потери, возникшие в результате использования бота. Перед включением AUTO_TRADE настоятельно рекомендуется бэктестирование стратегии на исторических данных.
