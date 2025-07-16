# Financial time series forecasting with AI

## Journey in Financial Forecasting: From Failed Models to a New Paradigm

I embarked on a project to forecast financial time series, specifically predicting stock price movements. My goal was to classify the next day's price movement into one of three categories: **decline, flat, or rise**. Here's a chronicle of my journey through various models, frustrating roadblocks, and an eventual, unexpected conclusion.

#### Phase 1: The Initial Approach - A Classification Task

My first strategy was to build a robust classification model. I chose XGBoost as a strong baseline and a Transformer model for its prowess in handling sequential data.

**My toolkit included:**
* **Stock Data:** Sourced using `pykrx`.
* **News Data:** Gathered with `gnews` and processed for sentiment analysis using the `'tabularisai/multilingual-sentiment-analysis'` model.
* **Technical Indicators:** A suite of features generated with the `ta` library.
* **Loss Function:** To combat the natural imbalance between "rise," "decline," and "flat" days, I used `class_weights` within the `CrossEntropyLoss` function.

**The Initial Results:**
* **XGBoost Accuracy:** 43.5%
* **Transformer Accuracy:** 41.7%

These results were disheartening. With a random chance baseline of 33.3%, my models were performing only marginally better. My primary suspect was the classic **class imbalance problem**, which I thought I had addressed.

#### Phase 2: The Grind - A Barrage of Failed Attempts

Convinced I could solve this, I systematically tried a wide array of techniques to improve performance. Each attempt, however, ended in failure.

* **Data Augmentation:** Neither adding noise nor using SMOTE (Synthetic Minority Over-sampling Technique) yielded any improvement.
* **Model & Training Adjustments:**
    * Changing the classification threshold.
    * Varying the number of technical features (both more and less).
    * Switching the model architecture from a Transformer to an LSTM.
    * Replacing `CrossEntropyLoss` with `FocalLoss` to better handle hard-to-classify examples.
    * Implementing `EarlyStopping` to prevent overfitting.
    * Increasing the `dropout` value for regularization.
    * Tuning the learning rate (`LR`) and applying gradient clipping.

Nothing worked. The performance remained stubbornly low.

#### Phase 3: The Pivot - New Data & New Problems

Frustrated with classification, I wondered if I was framing the problem incorrectly.

**Attempt 1: Switch to Regression**
I changed the task from classifying direction to regressing the future price. The result? An $RMSE$ of 0.0212. Impressively... it was even worse. This confirmed that predicting the exact price is significantly harder, so I rolled back to classification.

**Attempt 2: Add Macroeconomic Data**
Perhaps my model was missing the bigger picture. I integrated macroeconomic data from FRED:
* KOSPI Index
* KRW/USD Exchange Rate
* 10-Year Treasury Constant Maturity Rate

The result? Still no significant improvement. The "WHY????" echoed in my mind.

#### Phase 4: Exploring the State-of-the-Art (SOTA)

I decided it was time to see what cutting-edge research models could do. I looked into methods like 'Time-R1' and 'TiRex'. I started with TiRex.

Since TiRex doesn't work well in a non-CUDA Windows environment, I fired up a Google Colab notebook with a GPU. I fed it my data for a **zero-shot** prediction—meaning the model had *zero* training on my specific dataset.

The results were shocking. TiRex's zero-shot performance was substantially better than my meticulously trained custom models.

This was a major turning point. The idea of painstakingly building my own dataset and fine-tuning a model like Qwen2 4B (similar to the Fin-R1 approach) suddenly seemed inefficient.

#### The Final Revelation: Why Reinvent the Wheel?

As I contemplated the immense effort of fine-tuning, a simpler question emerged: "Why don't I just use an API?"

My research led me to a fascinating article by Kakao Bank, "[ChatGPT and Stock Price](https://tech.kakaobank.com/posts/2403-chatgpt-and-stock-price/)," which demonstrated the impressive capability of large language models (LLMs) like GPT-4 in financial contexts.

**Conclusion:**
My journey through the weeds of custom model building led me to a powerful conclusion. The challenge of stock market forecasting isn't just a numerical time-series problem; it's deeply intertwined with understanding the narrative, sentiment, and complex interplay of news and macroeconomic events.

This is precisely where massive, pre-trained LLMs excel. Instead of building a specialized model from scratch, the more effective and efficient path forward may be to leverage the emergent reasoning capabilities of models like GPT-4 via an API. My next steps will be to explore this new and promising direction.
