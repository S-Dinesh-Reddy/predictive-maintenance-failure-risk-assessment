import urllib.request
import os
import pandas as pd

target_path = os.path.join('data', 'ai4i2020.csv')

urls = [
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv',
    'https://raw.githubusercontent.com/Edoar-do/AI4I-Predictive-Maintenance/master/ai4i2020.csv'
]

downloaded = False
for url in urls:
    try:
        print(f"Attempting download from: {url}")
        urllib.request.urlretrieve(url, target_path)
        df = pd.read_csv(target_path)
        if len(df) == 10000:
            print(f"Successfully downloaded ai4i2020.csv ({len(df)} rows, {len(df.columns)} cols).")
            downloaded = True
            break
    except Exception as e:
        print(f"Failed from {url}: {e}")

if not downloaded:
    print("Download failed from all sources.")
