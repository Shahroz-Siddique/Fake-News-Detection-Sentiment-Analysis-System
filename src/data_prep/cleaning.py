import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')
fake_path=pd.read_csv('/home/foxtech/SHAHROZ_PROJ/Fake_news/data/Fake.csv')
ture_path=pd.read_csv('/home/foxtech/SHAHROZ_PROJ/Fake_news/data/True.csv')
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def load_and_prepare_data(fake_path, true_path, output_path):
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    df_fake['label'] = 0
    df_true['label'] = 1
    df = pd.concat([df_fake, df_true], axis=0).sample(frac=1).reset_index(drop=True)
    df['clean_content'] = df['text'].apply(clean_text)
    df.to_csv(output_path, index=False)
    return df
