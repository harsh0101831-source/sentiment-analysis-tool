import os
import re
import tempfile
from io import StringIO

import nltk
import pandas as pd
from flask import Flask, render_template, request, send_file
from nltk.corpus import stopwords
from textblob import TextBlob


nltk.download('stopwords')

app = Flask(__name__)
app.static_folder = 'static'




def clean_tweet(tweet):
    return ' '.join(
        re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(tweet)).split()
    )


def get_tweet_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity == 0:
        return "neutral"
    else:
        return "negative"


def preprocess_text(tweet):
    tweet = str(tweet).lower()
    tweet = re.sub(r'[^\w\s]', '', tweet)
    stop_words = set(stopwords.words('english'))
    tweet = ' '.join([word for word in tweet.split() if word not in stop_words])
    return tweet


def get_polarity_words(tweet):
    analysis = TextBlob(str(tweet))
    words = analysis.words
    word_polarities = {word: TextBlob(word).sentiment.polarity for word in words}
    sorted_words = sorted(word_polarities.items(), key=lambda item: item[1], reverse=True)
    top_words = [word for word, polarity in sorted_words[:5]]
    return ', '.join(top_words), analysis.sentiment.polarity


def get_tweets_from_csv(file_content):
    """
    SIMPLE VERSION:
    - Uses the 1st column as text
    - Uses the 2nd column (if exists) as date/time
    Works with almost any CSV.
    """
    df = pd.read_csv(StringIO(file_content))

    
    print("DEBUG - CSV columns:", df.columns.tolist())
    print("DEBUG - First 3 rows:\n", df.head())

    if df.empty:
        return []

    text_col = df.columns[0]
    date_col = df.columns[1] if len(df.columns) > 1 else None

    tweets = []

    for _, row in df.iterrows():
        content = row[text_col]
        date_val = row[date_col] if date_col is not None else ""

        sentiment = get_tweet_sentiment(content)
        preproc = preprocess_text(content)
        top_words, polarity_score = get_polarity_words(clean_tweet(content))

        tweets.append({
            "content": str(content),
            "preprocessed_content": preproc,
            "sentiment": sentiment,
            "date_time": str(date_val),
            "top_polarity_words": top_words,
            "polarity_score": polarity_score
        })

    return tweets




@app.route("/")
def home():
    return render_template("features.html")



@app.route("/predict", methods=["POST"])
def pred():
    try:
        if "csv_file" not in request.files:
            return render_template("result.html", result=None,
                                   csv_download=False, temp_file_path=None)

        csv_file = request.files["csv_file"]

        if csv_file.filename == "":
            return render_template("result.html", result=None,
                                   csv_download=False, temp_file_path=None)

        
        file_content = csv_file.read().decode("utf-8", errors="replace")

        print("DEBUG - Uploaded file name:", csv_file.filename)

        tweets = get_tweets_from_csv(file_content)

        print("DEBUG - Number of tweets parsed:", len(tweets))

        if not tweets:
            return render_template("result.html", result=None,
                                   csv_download=False, temp_file_path=None)

        
        df_out = pd.DataFrame(tweets)[
            ["content", "preprocessed_content", "sentiment", "date_time"]
        ]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_out.to_csv(temp_file.name, index=False, encoding="utf-8")
        temp_file_path = temp_file.name
        temp_file.close()

        return render_template("result.html",
                               result=tweets,
                               csv_download=True,
                               temp_file_path=temp_file_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return render_template("result.html", result=None,
                               csv_download=False, temp_file_path=None)



@app.route("/download_csv")
def download_csv():
    temp_file_path = request.args.get("temp_file_path")

    if not temp_file_path or not os.path.exists(temp_file_path):
        return "File not found", 404

    return send_file(
        temp_file_path,
        as_attachment=True,
        download_name="processed_results.csv"
    )



@app.route("/predict1", methods=["POST"])
def predict1():
    try:
        txt = request.form.get("txt", "").strip()

        if not txt:
            return render_template("result1.html",
                                   msg="No text entered.",
                                   result="N/A")

        sentiment = get_tweet_sentiment(txt)
        return render_template("result1.html",
                               msg=txt,
                               result=sentiment)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("result1.html",
                               msg="Error occurred.",
                               result=str(e))


if __name__ == "__main__":
    app.run(debug=True)
