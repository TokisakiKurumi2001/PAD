from keybert import KeyBERT
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

if __name__ == "__main__":
    # set seed
    random.seed(42)
    # number of keywords
    alpha = 0.35
    # load model
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    # double data
    df = pd.read_csv('pawsx.train.csv')
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

    blk_kws = []
    for sent in tqdm(new_sent2):
        blk = []
        no_words = len(sent.split(" "))
        keywords = [k[0] for k in kw_model.extract_keywords(sent, top_n=int(no_words * alpha))]
        blk_kws.append("|".join(keywords))

    length = len(new_label.tolist())
    tasks = ["MASK"] * length
    tasks += ["CLS"] * length

    # increase the examples since we have 2 tasks, one for masking, the other for classification (purely)
    new_sent1 = np.hstack([new_sent1, new_sent1])
    new_sent2 = np.hstack([new_sent2, new_sent2])
    new_label = np.hstack([new_label,new_label])
    new_lang = np.hstack([new_lang,new_lang])
    blk_kws += [""] * length

    new_df = pd.DataFrame({'sentence1': new_sent1, 'sentence2': new_sent2, 'label': new_label, 'lang': new_lang, 'blk_kws': blk_kws, "task": tasks})
    new_df.to_csv('pawsx-extend.train.csv', index=False)
    