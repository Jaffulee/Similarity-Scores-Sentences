# Need below dependencies
# from sentence_transformers import SentenceTransformer, util
# import numpy as np
# import re
import pandas as pd
from Word_similarity import do_it_all_similarity_df_out_both

if __name__ == '__main__':
    csv_path = r"Pub Names\pub_names.csv"
    df = pd.read_csv(csv_path,
                    encoding="latin1",     # or "cp1252"
                    sep=",",               # keep default comma separator
                    engine="python")       # more forgiving parser 
    csv_out = r'Pub Names\pub_names_similarity_scores.csv'
    df = (
        df
        .drop_duplicates(['Name'])
        .reset_index()
        # .head(100)
          )
    # print(df)
    df = do_it_all_similarity_df_out_both(df['Name'],columnnames=['Pub_Name_1','Pub_Name_2','Pub_Name_1_ID','Pub_Name_2_ID','Similarity_Score_Sentencewise','Similarity_Score_Wordwise'])
    df.to_csv(csv_out, index=False, encoding="utf-8")
    print('Saved to',csv_out)