import pandas as pd
import numpy as np
import random
from tqdm import tqdm

if __name__ == "__main__":
    # set seed
    random.seed(42)
    # double data
    df = pd.read_csv('pawsx.test.csv')
    df = df.drop(['id'], axis=1)

    # swap sentence 1 and 2
    sent1 = df['sentence1'].values
    sent2 = df['sentence2'].values
    new_sent1 = np.hstack([sent1,sent2])
    new_sent2 = np.hstack([sent2,sent1])

    # increase other columns
    label = df['label'].values
    new_label = np.hstack([label,label])
    lang = df['lang'].values
    new_lang = np.hstack([lang,lang])

    length = len(new_label.tolist())
    blk_kws = [""] * length
    tasks = ["CLS"] * length

    new_df = pd.DataFrame({'sentence1': new_sent1, 'sentence2': new_sent2, 'label': new_label, 'lang': new_lang, 'blk_kws': blk_kws, "task": tasks})
    new_df.to_csv('pawsx-extend.test.csv', index=False)
    