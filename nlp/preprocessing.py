import pandas as pd
import numpy as np
import os
import re
import nltk
import string


def get_word_stats(df, col_name, nltk_tokenizer=False):
    column = df[col_name]

    word_stats = []
    sentences = []
#     avg_word_len = []

    if nltk_tokenizer:
        for doc in column:
            text = nltk.sent_tokenize(doc)
            words = [len(sen.split()) for sen in text]
            
            word_len = [len(word) for sentence in text for word in sentence.split()]
            word_avg = np.mean(word_len)
            
            word_stats.append((sum(words), len(words), np.mean(words), np.max(words), np.min(words), word_avg))
            sentences.append(text)
    else:
        for doc in column:
            text = remove_newline(doc)
            split_text = re.split('[.?!]', text)
            split_text = [sen for sen in split_text if len(sen) > 2]
            words = [len(sen.split()) for sen in split_text]
            
            word_len = [len(word) for sentence in split_text for word in sentence.split()]
            word_avg = np.mean(word_len)
            
            word_stats.append((sum(words), len(words), np.mean(words), np.max(words), np.min(words), word_avg))
            sentences.append(split_text)

    return word_stats, sentences

def get_avg_word_length(df, col_name, nltk_tokenizer=True):
    stats, sen = get_word_stats(df, col_name, nltk_tokenizer=True)
    avg_word = []
    for stat in stats:
        avg_word.append(round(stat[-1], 2))
    
    return avg_word


def get_sentences(df, text_col, target_col):
    """Returns tuple of each document split by sentence with additional info:
    (sentence, number of words, average length of words, cefr_level, document id)

    Parameters
    ----------
    A dataframe: Pandas dataframe
    Name of text or document column: str
    Name of target column: str

    Returns
    -------
    A tuple
    (See Info Above)
    """
    sentences = []
    for row in df.itertuples():
        text = nltk.sent_tokenize(getattr(row, text_col).strip())
        level = getattr(row, target_col)
        doc = getattr(row, 'Index')
        for sentence in text:
            words = remove_punct(sentence).split()
            if len(words) == 0:
                avg_len_words = 0
            else:
                avg_len_words = sum(len(word) for word in words if len(word) > 0)/len(words)
            sentences.append((sentence, len(words), round(avg_len_words, 2) , level, doc))

    return sentences

def split_into_sentence(docs, levels):
    """Returns tuple of each document broken down into sentences and level. """
    sentences = []
    doc_id = 0
    for i in range(len(levels)):
        doc = nltk.sent_tokenize(docs[i])
        level = levels[i]
        for sentence in doc:
            sentences.append((sentence, len(sentence.split()), level, doc_id))
        doc_id += 1
    return sentences

def get_sentences_dataframe(df, col_text, col_level):
    records = split_into_sentence(df[col_text], df[col_level])
    
    return pd.DataFrame.from_records(records, columns=['documents', 'num_of_words', 'level', 'doc_id'])

def get_empty_grammar_dictionary(df, col_name):
    grammar_dictionary = {}
    for doc in df[col_name]:
        pos = nltk.pos_tag_sents([doc.split()])
        for pair in pos[0]:
            if pair[1] not in grammar_dictionary.keys():
                grammar_dictionary[pair[1]] = 0
            else:
                pass
    return grammar_dictionary

def get_part_of_speech_dictionary(df, col_name):
    gd = get_empty_grammar_dictionary(df, col_name)
    grammar_dictionary = gd.copy()
    pos_list = []
    grammar_dict = {}
    for doc in df['documents']:
        grammar_dict = grammar_dict.fromkeys(grammar_dictionary, 0)
        pos = nltk.pos_tag_sents([doc.split()])
        for pair in pos[0]:
            grammar_dict[pair[1]] += 1
        
        pos_list.append(grammar_dict)
    
    return pos_list

def get_part_of_speech_dataframe(df, col_name):
    
    return pd.DataFrame.from_dict(get_part_of_speech_dictionary(df, col_name))


def get_word_level_dataframe(data, col_name, vocab_dict, adv_vocab_dict):
    
    return pd.DataFrame.from_records(get_word_level_dictonaries(data, col_name, vocab_dict, adv_vocab_dict))



def get_full_dataframe(df, col_text, col_level, levels_df, pos_df):
    data = df.copy()
    df_1 = get_sentences_dataframe(df, col_text, col_level)
    total_words = df_1.groupby('doc_id').sum()
    data = data.join(total_words)
#     levels_df = pd.DataFrame.from_records(level_counts)
    data = data.join(levels_df)
    data = data.join(pos_df)
    
    return data
    
    


def get_word_level_dictonaries(df, col_name, vocab_dict, adv_vocab_dict):
    words = get_tokenized_words_treebank(df, col_name)
    level_counts = []
    for doc in words:
        level_dict = {'A1': 0, 'A2': 0, 'B1': 0, 'B2': 0, 'C1': 0, 'C2': 0}
        for sentence in doc:
            for word in sentence:
                a1_upper = ['I', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 
                            'November', 'December', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

                if word in a1_upper:
                    level_dict['A1'] += 1
                elif word in ['.', ',', '!', '?']:
                    pass
                else:
                    original_word = word
                    word = word.strip('"')
                    try:
                        level = vocab_dict[word.lower()]
                        level_dict[level] += 1
                    except KeyError:
                        try:
                            word = word[:-1]
                            level = vocab_dict[word]
                            level_dict[level] += 1

                        except KeyError:
                            try:
                                word = word[:-1]
                                level = vocab_dict[word]
                                level_dict[level] += 1
                            except KeyError:
                                try:
                                    word = word[:-1]
                                    level = vocab_dict[word]
                                    level_dict[level] += 1
                                except KeyError:
                                    try:
                                        level = adv_vocab_dict[original_word]
                                        level_dict[level] += 1
                                    except KeyError:
                                        pass
        level_counts.append(level_dict)
    
    return level_counts


        
#     for row in df.itertuples():
#         text = nltk.sent_tokenize(getattr(row, text_col).strip())
#         level = getattr(row, target_col)
#         doc = getattr(row, 'Index')
#         for sentence in text:
#             words = remove_punct(sentence).split()
#             avg_len_words = sum(len(word) for word in words)/len(words)
#             sentences.append((sentence, len(words), round(avg_len_words, 2) , level, doc))

#     return sentences

def process_sample(text, list_of_keys, vocab_dict, adv_vocab_dict):
    level_dict = {'A1': 0, 'A2': 0, 'B1': 0, 'B2': 0, 'C1': 0, 'C2': 0}
    sentence = text.split()
    for word in sentence:
        a1_upper = ['I', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 
                    'November', 'December', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        if word in a1_upper:
            level_dict['A1'] += 1
        elif word in ['.', ',', '!', '?']:
            pass
        else:
            original_word = word
            word = word.strip('"')
            try:
                level = vocab_dict[word.lower()]
                level_dict[level] += 1
            except KeyError:
                try:
                    word = word[:-1]
                    level = vocab_dict[word]
                    level_dict[level] += 1

                except KeyError:
                    try:
                        word = word[:-1]
                        level = vocab_dict[word]
                        level_dict[level] += 1
                    except KeyError:
                        try:
                            word = word[:-1]
                            level = vocab_dict[word]
                            level_dict[level] += 1
                        except KeyError:
                            try:
                                level = adv_vocab_dict[original_word]
                                level_dict[level] += 1
                            except KeyError:
                                pass

    pos_list = []
    grammar_dict = dict.fromkeys(list_of_keys, 0)
    grammar_dict

    pos = nltk.pos_tag_sents([sentence])
    for pair in pos[0]:
        try:
            grammar_dict[pair[1]] += 1
        except KeyError:
            pass
    
    def merge(dict1, dict2):
        new_dict = {**dict1, **dict2}
    
        return new_dict
    
    sample_record = merge(level_dict, grammar_dict)
    sample = pd.DataFrame.from_dict([sample_record])
    sample.insert(0, 'num_of_words', len(sentence))
    text = remove_punct(text)
    words = text.split()
    word_len = [len(word) for word in words]
    sample.insert(0, 'avg_word_len', np.mean(word_len))
    
    return sample.loc[0]


def tokenize_sentences(df, col_name):
    column = df[col_name]

    sentences = []

    for doc in column:
        text = nltk.sent_tokenize(doc.strip())
        sentences.append(text)

    return sentences


def tokenize_words(list_of_sentences, tokenizer_type='regexp'):
    if tokenizer_type == 'regexp':
        tokenizer = nltk.RegexpTokenizer(r"\w+")
    elif tokenizer_type == 'treebank':
        tokenizer = nltk.TreebankWordTokenizer()
    else:
        raise Exception("Unrecognized keyword arguement. Must be 'regexp' or 'treebank'. Note: 'regexp' is the default")

    document_word_list = []
    for doc in list_of_sentences:

        word_list = []
        for sen in doc:
            word_list.append(tokenizer.tokenize(sen))
        
        document_word_list.append(word_list)

    return document_word_list


def tokenized_word_stats(list_of_words):
    word_stats = []
    for doc in list_of_words:
        
        len_of_sents = []
        for word_list in doc:
            len_of_sents.append(len(word_list))
        
        word_stats.append(
            (sum(len_of_sents), len(len_of_sents), np.mean(len_of_sents), np.max(len_of_sents), np.min(len_of_sents)))

    return word_stats


def parts_of_speech(list_of_words):
    pos_list = []
    for doc in list_of_words:
        pos_list.append(nltk.pos_tag_sents(doc))

    return pos_list


def get_tokenized_word_stats(df, col_name):
    """Returns Total number of Words, Total Number of Sentences, Average Sentence Length, Maximum and Minimum Sentence lengths"""
    
    return tokenized_word_stats(tokenize_words(tokenize_sentences(df, col_name)))

# def get_tokenized_word_stats_treebank(df, col_name):
#     """Returns Total number of Words, Total Number of Sentences, Average Sentence Length, Maximum and Minimum Sentence lengths"""
    
#     return tokenized_word_stats(tokenize_words(tokenize_sentences(df, col_name)))


def get_tokenized_words_treebank(df, col_name):
    
    return tokenize_words(tokenize_sentences(df, col_name), tokenizer_type='treebank')


def get_tokenized_words(df, col_name):
    
    return tokenize_words(tokenize_sentences(df, col_name))


def get_tokenized_sentences(df, col_name):
    
    return tokenize_sentences(df, col_name)


def get_all_tokenized_lists(df, col_name):
    
    return get_tokenized_sentences(df, col_name), get_tokenized_words(df, col_name), get_tokenized_word_stats(df,
                                                                                                              col_name)
# Remove noise: remove digits, unwanted lines like email lines, and excess whitespace.
def remove_noise(text):
    # Remove any numbers
    text = re.sub("^\d+\S+|\S+\d+\S+|\s+\d+\s+|\S+\d+$", "", text)

    # Remove any words that contain punctuation exluding proper punctuation endings
    text= re.sub('\S+[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]\S+', " ", text)
    text =re.sub('\s[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]\s+', " ", text)
    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]\S+', " ", text)

    # Exclude , . ! ? ; : -
    text = re.sub('\S+["#$%&\'()*+/<=>@[\\]^_`{|}~]', " ", text)

    # Remove any excess whitespace
    text = re.sub('\s+', " ", text)
    
    return text.strip()


# Remove new line character
def remove_newline(text):
    text = "".join([char for char in text if char not in '\n'])

    return text.strip()


# Remove punctuation
from nltk.tokenize import word_tokenize

def remove_punct(text):
    
#     new_text = word_tokenize(text)
#     new_text = list(filter(lambda token: token not in string.punctuation, new_text))
#     text = " ".join([word for word in new_text])    
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    new_text = tokenizer.tokenize(text)
    text = " ".join([word for word in new_text])

    return text.strip()


# Convert to lowercase
def to_lower(text):
    text = "".join([char.lower() for char in text])

    return text.strip()


# Remove stop words for English
stopwords = nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    words = text.split()
    final_text = " ".join([word for word in [word for word in text.split() if word not in stopwords]])

    return final_text


# Lemmatizer
from nltk.stem import WordNetLemmatizer


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    new_text = [lemmatizer.lemmatize(word) for word in words]
    final_text = " ".join([word for word in new_text])

    return final_text


# Porter Stemming
from nltk.stem import PorterStemmer


def stemmed(text):
    ps = PorterStemmer()

    words = text.split()

    new_text = [ps.stem(word) for word in words]
    final_text = " ".join([word for word in new_text])
    
    return final_text

# Deprecated: this functionality was used in original text cleaning
def replace_apostrophe(text):
    text = re.sub("`|’", "'", text)
    
    return text.strip()


def preprocess(df, col_name, punct=True, lower=True, stop_words=True, noise=True):
    """Returns a new column for a dataframe with all following preprocessed steps: removes punctuation based on python
    string, converts all characters to lower case, and removes all the stopwords based on nltk's stopwords.
    
    Option to toggle off any of the three types of preprocessing. All punct, lower, and stop_words defaulted to True.

    Parameters
    ----------
    A dataframe: Pandas dataframe
    Name of the column: str

    Returns
    -------
    A Pandas series for a new column in a data frame
    """
    column = df[col_name]
    
    if noise:
        column = column.apply(lambda x: remove_noise(x))
    if punct:
        column = column.apply(lambda x: remove_punct(x))
    if lower:
        column = column.apply(lambda x: to_lower(x))
    if stop_words:
        column = column.apply(lambda x: remove_stopwords(x))

    return column


def preprocess_all(df, col_name, punct=True, lower=True, stop_words=True, noise=True):
    """Returns new dataframe with all the preprocessing from the preprocess function with to lemmitizing and stemming.

    Parameters
    ----------
    A dataframe: Pandas dataframe
    Name of the column: str

    Returns
    -------
    A new data frame with fully, preprocessed column
    """
    dataframe = df.copy()
    new_col_name = col_name + "_clean"
    dataframe[new_col_name] = preprocess(df, col_name, punct, lower, stop_words, noise)

    dataframe['Lemmatized'] = dataframe[new_col_name].apply(lambda x: lemmatization(x))
    dataframe['Stemmed'] = dataframe[new_col_name].apply(lambda x: stemmed(x))

    return dataframe

def get_cefr_grammar_dictionary_adv(path='data/octanove-vocabulary-profile-c1c2-1.0.csv'):
    vocab = pd.read_csv(path)
    grammar_dict = {}
    for row in vocab.itertuples():
        word = getattr(row, 'headword')
        pos = getattr(row, 'pos')
        if re.search('/', word):
            word = word.split('/')
        if type(word) is str:
            grammar_dict[word] = pos
        else:
            for w in word:
                grammar_dict[w] = pos
    
    return grammar_dict



def get_cefr_word_dictionary_adv(path='data/octanove-vocabulary-profile-c1c2-1.0.csv'):
    vocab = pd.read_csv(path)
    vocab_dict = {}
    for row in vocab.itertuples():
        word = getattr(row, 'headword')
        level = getattr(row, 'CEFR')
        if re.search('/', word):
            word = word.split('/')
        if type(word) is str:
            vocab_dict[word] = level
        else:
            for w in word:
                vocab_dict[w] = level
    
    return vocab_dict

def get_cefr_grammar_dictionary(path='data/cefrj-vocabulary-profile-1.5.csv'):
    vocab = pd.read_csv(path)
    grammar_dict = {}
    for row in vocab.itertuples():
        word = getattr(row, 'headword')
        pos = getattr(row, 'CEFR')
        if re.search('/', word):
            word = word.split('/')
        if type(word) is str:
            grammar_dict[word] = pos
        else:
            for w in word:
                grammar_dict[w] = pos
    
    extras_a1 = ["n't", "'ve'","'ll", "making", "taking", "putting", "sitting", "cutting", 'been', 'begun', 'chosen', 'come', 'done', 
                 'driven', 'eaten', 'felt', 'found', 'flown', 'got', 'gotten', 'given', 'gone', 'had', 'heard', 'known', 'left', 'made', 'met', 
                 'paid', 'put', 'read', 'run', 'said', 'seen', 'sent', 'sung', 'sat', 'slept', 'spoken', 'swum', 'taken', 'taught', 'told', 
                 'thought', 'understood', 'worn', 'written', 'gave', 'bought', 'families', 'stories', 'cities', 'died', 'looking',
                 'studied', 'having', 'wrote', 'travelling', 'countries', 'giving', 'knew', 'getting', 'watching', 'drunk']
    for word in extras_a1:
        grammar_dict[word] = 'A1'
    
    extras_a2 = ['become', 'brought', 'built', 'fallen', 'forgotten', 'grew', 'kept', 'lent', 'let', 'lost', 'sold', 'stood', 'dug', 
                 'became', 'cried', 'proving', 'flew', 'busiest', 'threw', 'wettest', 'arriving', 'mistaking', 'continuing', 'racing',
                'providing', 'centuries', 'fell', 'shading']
    for word in extras_a2:
        grammar_dict[word] = 'A2'
        

    return grammar_dict



def get_cefr_word_dictionary(path='data/cefrj-vocabulary-profile-1.5.csv'):
    vocab = pd.read_csv(path)
    vocab_dict = {}
    for row in vocab.itertuples():
        word = getattr(row, 'headword')
        level = getattr(row, 'CEFR')
        if re.search('/', word):
            word = word.split('/')
        if type(word) is str:
            vocab_dict[word] = level
        else:
            for w in word:
                vocab_dict[w] = level
    
    extras_a1 = ["n't", "'ve'","'ll", "making", "taking", "putting", "sitting", "cutting", 'been', 'begun', 'chosen', 'come', 'done', 
                 'driven', 'eaten', 'felt', 'found', 'flown', 'got', 'gotten', 'given', 'gone', 'had', 'heard', 'known', 'left', 'made', 'met', 
                 'paid', 'put', 'read', 'run', 'said', 'seen', 'sent', 'sung', 'sat', 'slept', 'spoken', 'swum', 'taken', 'taught', 'told', 
                 'thought', 'understood', 'worn', 'written', 'gave', 'bought', 'families', 'stories', 'cities', 'died', 'looking',
                 'studied', 'having', 'wrote', 'travelling', 'countries', 'giving', 'knew', 'getting', 'watching', 'drunk']
    for word in extras_a1:
        vocab_dict[word] = 'A1'
    
    extras_a2 = ['become', 'brought', 'built', 'fallen', 'forgotten', 'grew', 'kept', 'lent', 'let', 'lost', 'sold', 'stood', 'dug', 
                 'became', 'cried', 'proving', 'flew', 'busiest', 'threw', 'wettest', 'arriving', 'mistaking', 'continuing', 'racing',
                'providing', 'centuries', 'fell', 'shading']
    for word in extras_a2:
        vocab_dict[word] = 'A2'
        

    return vocab_dict


def simple_clean_punctuation(df, col_name, new_line=False):
    """Returns Pandas dataframe with punctuation removed from selected column, with the option to remove all new lines.

    Parameters
    ----------
    A dataframe: Pandas dataframe
    Name of the column: str

    Returns
    -------
    A new Pandas Data Frame
    """
    df_new = df.copy()
    col = df_new[col_name]
    new_doc = []
    for doc in col:
        if new_line:
            pass
        else:
            doc = doc.replace('\n', ' ')
        new = ''
        for char in doc:
            if char not in string.punctuation:
                new = "".join((new, char))
        new_doc.append(new)
    df_new[col_name] = new_doc

    return df_new


# Percentage of True Target values
def percentage_of_target(df, target):
    print(f'Percentage of Target that are True: {(sum(df[target] == 1) / df[target].shape[0]) * 100}%')
