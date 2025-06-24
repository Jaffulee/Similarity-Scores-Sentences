from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import re
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")   # small, 384-d
print("model loaded\n")


def test_word_sim_by_position(emb,pos1,pos2):
    return util.cos_sim(emb[pos1], emb[pos2]).item()

def test_word_sim_by_positions(emb,pos1,pos2s):
    result = []
    for pos2 in pos2s:
        result.append(test_word_sim_by_position(emb,pos1,pos2))
    return result

def test_words_sim_by_positions_agg(emb,pos1s,pos2s):
    test_results = []
    for pos1 in pos1s:
        test_results.append(np.max(test_word_sim_by_positions(emb,pos1,pos2s)))
    return np.average(test_results)

def get_list_of_compared_sentences_wordwise(arrbig,poss):
    compared_sentences_arr = []
    for i,posi in enumerate(poss):
        compared_sentences_row = []
        for j, posj in enumerate(poss):
            if i==j:
                compared_sentences_row.append(1)
            elif j<i:
                compared_sentences_row.append(compared_sentences_arr[j][i])
            else:
                compared_sentences_row.append(test_words_sim_by_positions_agg(arrbig,posi,posj))
        compared_sentences_arr.append(compared_sentences_row)
    return compared_sentences_arr


def get_list_of_compared_sentences_sentencewise(emb):
    compared_sentences_arr = []
    for i,itemi in enumerate(emb):
        compared_sentences_row = []
        for j, itemj in enumerate(emb):
            if i==j:
                compared_sentences_row.append(1)
            elif j<i:
                compared_sentences_row.append(compared_sentences_arr[j][i])
            else:
                compared_sentences_row.append(util.cos_sim(emb[i], emb[j]).item())
        compared_sentences_arr.append(compared_sentences_row)
    return compared_sentences_arr



def convert_sentence_to_array_words(sentence,additional_matching_str = None):
    pattern = r'\s+'
    if additional_matching_str !=None:
        pattern+='|'+additional_matching_str
    pattern_symbols = r'\W'
    sentence = re.sub(pattern_symbols,r' ',sentence)
    return re.split(pattern,sentence)

def convert_array_of_sentences_to_words(sentence_arr,additional_matching_str = None):
    return [convert_sentence_to_array_words(x,additional_matching_str) for x in sentence_arr]

def convert_array_of_arrays_to_big_array(arrofarr):
    arrbig = []
    pos_new = 0
    pos_of_original_arr = 0
    poss_news = []
    

    for pos_of_original_arr, x in enumerate(arrofarr):
        poss_new = []
        for y in x:
            arrbig.append(y)
            poss_new.append(pos_new)
            pos_new+=1
        poss_news.append(poss_new)
    return arrbig,poss_news

def convert_nxn_array_to_long_df(nxnarr,columnnames = ['i','j','value']):
    column1 = []
    column2 = []
    values = []
    for i,row in enumerate(nxnarr):
        for j,value in enumerate(row):
            column1.append(i)
            column2.append(j)
            values.append(value)
    df = pd.DataFrame({
        columnnames[0]: column1,
        columnnames[1]: column2,
        columnnames[2]: values
    })
    return df

def attach_sentences_to_long_sim_df(sentence_arr,nxnarr,columnnames = ['Sentence_1','Sentence_2','Sentence_1_ID','Sentence_2_ID','Similarity_Score']):
    sentence_df_1 = pd.DataFrame({
        columnnames[0]: sentence_arr
    })
    sentence_df_2 = pd.DataFrame({
        columnnames[1]: sentence_arr
    })
    sentence_df_1['Sentence_ID'] = sentence_df_1.index
    sentence_df_2['Sentence_ID'] = sentence_df_2.index
    long_sim_df = convert_nxn_array_to_long_df(nxnarr,columnnames[2:])
    df_out = (long_sim_df
              .merge(sentence_df_1,how='inner',left_on=columnnames[2],right_on='Sentence_ID')
              .merge(sentence_df_2,how='inner',left_on=columnnames[3],right_on='Sentence_ID')
              )
    del df_out['Sentence_ID_x'], df_out['Sentence_ID_y']
    return df_out

def do_it_all_similarity_df_out_wordwise(sentence_arr,columnnames = ['Sentence_1','Sentence_2','Sentence_1_ID','Sentence_2_ID','Similarity_Score']):
    sentence_arr_arr = convert_array_of_sentences_to_words(sentence_arr)
    sentence_bigarr, poss = convert_array_of_arrays_to_big_array(sentence_arr_arr)
    emb  = model.encode(sentence_bigarr, convert_to_tensor=True, normalize_embeddings=True)
    print('df_w loaded')
    result = get_list_of_compared_sentences_wordwise(emb,poss)
    result = attach_sentences_to_long_sim_df(sentence_arr,result,columnnames)
    return result

def do_it_all_similarity_df_out_sentencewise(sentence_arr,columnnames = ['Sentence_1','Sentence_2','Sentence_1_ID','Sentence_2_ID','Similarity_Score']):
    result = get_list_of_compared_sentences_sentencewise(model.encode(sentence_arr, convert_to_tensor=True, normalize_embeddings=True))
    # result2 = convert_nxn_array_to_long_df(result2,['Sentence 1','Sentence 2','Score'])
    result = attach_sentences_to_long_sim_df(sentence_arr,result,columnnames)
    return result

def do_it_all_similarity_df_out_both(sentence_arr,columnnames = ['Sentence_1','Sentence_2','Sentence_1_ID','Sentence_2_ID','Similarity_Score_Sentencewise','Similarity_Score_Wordwise']):
    columnnames_wordwise = columnnames
    valuename_s = columnnames_wordwise.pop(-2)


    df_s = get_list_of_compared_sentences_sentencewise(model.encode(sentence_arr, convert_to_tensor=True, normalize_embeddings=True))
    df_s = convert_nxn_array_to_long_df(df_s,['Dummy 1','Dummy 2',valuename_s])

    print('df_s loaded')
    df_w = do_it_all_similarity_df_out_wordwise(sentence_arr,columnnames_wordwise)
    df_out = df_w.merge(df_s,left_on=[columnnames_wordwise[2],columnnames_wordwise[3]], right_on = ['Dummy 1', 'Dummy 2'])
    del df_out['Dummy 1'], df_out['Dummy 2']
    return df_out

if __name__ == '__main__':
    # Test 1: simple words
    words = ["cat", "dog", "happy"] 
    emb  = model.encode(words, convert_to_tensor=True, normalize_embeddings=True)

    sim_cat_dog   = util.cos_sim(emb[0], emb[1]).item()
    sim_cat_happy = util.cos_sim(emb[0], emb[2]).item()

    print("cat–dog:", sim_cat_dog)
    print("cat–happy:", sim_cat_happy)
    
    # Test 2: array of sentences 1
    Sentence_1 = 'also The Witch & Sow'
    print(convert_sentence_to_array_words(Sentence_1))

    Sentence_2 = 'Moorgate, Hillingsdon and Lewisham Arms'
    Sentence_3 = 'And The Wizard & Reap'
    sentence_arr = [Sentence_1,Sentence_2, Sentence_3]

    # Usage
    sentence_arr2 = convert_array_of_sentences_to_words(sentence_arr)
    sentence_bigarr, poss = convert_array_of_arrays_to_big_array(sentence_arr2)
    words = sentence_bigarr
    emb  = model.encode(words, convert_to_tensor=True, normalize_embeddings=True)

    # Output wordwise
    result1 = get_list_of_compared_sentences_wordwise(emb,poss)
    # result1 = convert_nxn_array_to_long_df(result1,['Sentence 1','Sentence 2','Score'])
    result1 = attach_sentences_to_long_sim_df(sentence_arr,result1)
    print(result1)

    print(do_it_all_similarity_df_out_wordwise(sentence_arr))

    # Test 2.5: Output sentencewise
    result2 = get_list_of_compared_sentences_sentencewise(model.encode(sentence_arr, convert_to_tensor=True, normalize_embeddings=True))
    # result2 = convert_nxn_array_to_long_df(result2,['Sentence 1','Sentence 2','Score'])
    result2 = attach_sentences_to_long_sim_df(sentence_arr,result2)
    print(result2)
    print(do_it_all_similarity_df_out_sentencewise(sentence_arr))

    result3 = do_it_all_similarity_df_out_both(sentence_arr)
    del result3['Sentence_1'], result3['Sentence_2']
    print(result3)

    csv_path = r"Other\pub_names.csv"
    df = pd.read_csv(csv_path,
                    encoding="latin1",     # or "cp1252"
                    sep=",",               # keep default comma separator
                    engine="python")       # more forgiving parser 
    csv_out = r'Other\pub_names_similarity_scores.csv'
    df = df.drop_duplicates(['Name']).reset_index()
    print(df)
    df = do_it_all_similarity_df_out_both(df['Name'],columnnames=['Pub_Name_1','Pub_Name_2','Pub_Name_1_ID','Pub_Name_2_ID','Similarity_Score_Sentencewise','Similarity_Score_Wordwise'])
    df.to_csv(csv_out, index=False, encoding="utf-8")

    quote_list = [
        '''I'm quite put out!''',
        '''I am most seriously displeased!''',
        '''Mary wished to say something sensible, but knew not how.''',
        '''She is tolerable I suppose, but not handsome enough to tempt me''',
        '''Happy thought indeed.''',
        '''It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife''',
        '''I have not that talent of conversing easily with people I have never met''',
        '''You must allow me to tell you how ardently I admire and love you.''',
        '''I have been meditating on the very great pleasure which a pair of fine eyes in the face of a pretty woman can bestow.''',
    ]
    num_guesses = 15
    for quote in quote_list:
        print()
        quote_arr = [quote]
        guess_arr = quote_arr
        for guess_num in range(15):
            guess = input('Guess the quote! Type "Resign" to give up.\n')
            guess_arr.append(guess)
            if guess == 'Resign':
                print('\nThe quote was:\n',quote)
                if len(guess_arr)>0:
                    print('Your best guesses were')
                    print(df[(df['Similarity_Score_Wordwise'] == df['Similarity_Score_Wordwise'].max()) | (df['Similarity_Score_Sentencewise'] == df['Similarity_Score_Sentencewise'].max()) ])
                break
            df = do_it_all_similarity_df_out_both(guess_arr,columnnames = ['Quote','Guess','Sentence_1_ID','Sentence_2_ID','Similarity_Score_Sentencewise','Similarity_Score_Wordwise'])
            df = df[(df['Sentence_1_ID']==0) & (df['Sentence_2_ID']>0)]
            del df['Sentence_1_ID'], df['Sentence_2_ID'], df['Quote']
            print(df)
        if len(guess_arr) == num_guesses:
            print('\nOut of guesses! The quote was:\n',quote)
    
    
