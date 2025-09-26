import pandas as pd
import argparse

def fraction_type(x):
    try:
        value = int(x)
        if not (0 < value <= 100):
            raise argparse.ArgumentTypeError("Fraction must be an integer between 1 and 100.")
        return value / 100.0
    except ValueError:
        raise argparse.ArgumentTypeError("Fraction must be an integer between 1 and 100.")
parser = argparse.ArgumentParser()

parser.add_argument('fraction_percentage', type=fraction_type, help="Fraction of the dataset to keep (1-100)")
args = parser.parse_args()
fraction = args.fraction_percentage

df = pd.read_csv('creditcard.csv') #https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df.drop(columns=['Time'], inplace=True)
n_rows_keep = int(len(df) * fraction)
df = df.head(n_rows_keep).copy()

df.to_csv(f'creditcard_small_{int(fraction*100)}.csv', index=False)