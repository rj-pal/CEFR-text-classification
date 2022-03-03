import os
import io
import re
import pandas as pd
import nltk


# OneStopEnglishCorpus Related Functions

# Main directory for OneStopEnglishCorpus from https://github.com/nishkalavallabhi/OneStopEnglishCorpus
main_path = '../OneStopEnglishCorpus/Texts-SeparatedByReadingLevel/'

# Sub path list for documents containing individual levels
sub_paths = ['Ele-Txt/', 'Int-Txt/', 'Adv-Txt/']

def get_docs_one_stop(document_id=True):
    """Returns a list of sentences of the parsed documents and the associated level as a tuple
    with an option to include an original document_id tag to track the original document.
    
    Parameters
    ----------
    None (Optional doc_id): bol

    Returns
    -------
    A tuple 
    (1) A of a list of sentences of the parsed documents (text) from OneStopEnglishCorpus
    (2) Level of the text (Ele, Int, or Adv)
    (3) (Optional) original document_id tag 
    """
    docs = []
    doc_id = 0

    for path in sub_paths:
        directory_path = main_path + path
        directory = os.fsencode(directory_path)
        level = path[:3]
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith('.txt'):
                with io.open(directory_path + filename, encoding = 'utf-8-sig') as f:
                    document = f.read()
                    sents = document.split('\n')
                    sents = [sen for sen in sents if len(sen) > 0]
                    if document_id:
                        docs.append((sents, level, doc_id))
                    else:
                        docs.append((sents, level))
                doc_id += 1

    return docs

def split_docs_one_stop(document_tuple, document_id=True):
    """Returns a parsed string of the document sentences, the sentences in list-form, the average number of
    words per sentence, the total number of sentences, the total words in the segment, the number of words 
    in each sentence segment, and the associated level.
    
    Parameters
    ----------
    Document info of OneStopEnlishCorpus: tuple

    Returns
    -------
    A tuple for dataframe creation.
    (See Info Above)
    """
    seperated_doc = []
    for doc in document_tuple:
        level = doc[1]
        doc_id = doc[2]
        for segment in doc[0]:
            sentences = nltk.sent_tokenize(segment)
            num_of_sen = len(sentences)
            total_words = sum(len(sen.split()) for sen in sentences)
            sen_len_tuple = [len(sen.split()) for sen in sentences]
            avg_num_words = round(total_words/num_of_sen, 2)
            sent_string = " ".join([sen for sen in sentences])
            sent_string = sent_string.strip()
            if document_id:
                seperated_doc.append((sent_string, sentences, avg_num_words, num_of_sen, total_words, sen_len_tuple, level, doc_id))
            else:
                seperated_doc.append((sent_string, sentences, avg_num_words, num_of_sen, total_words, sen_len_tuple, level))
    
    return seperated_doc

def get_one_stop_dataframe(levels=True, document_id=True):
    """Returns the OneStopEnglishCorpus Dataframe. If levels = False, will return numberic levels:
    Ele = 0, Int =1, Adv = 2
    
    Parameters
    ----------
    None

    Returns
    -------
    A Pandas Data Frame  
    """
    records = split_docs_one_stop(get_docs_one_stop())
    
    if document_id:
        df = pd.DataFrame.from_records(records, columns=['documents', 'doc_list', 'avg_num_words', 'total_num_sents', 'total_num_words', 
                                                       'words_per_sents', 'level', 'doc_id'])
    else:
        df = pd.DataFrame.from_records(records, columns=['documents', 'doc_list', 'avg_num_words', 'total_num_sents', 'total_num_words', 
                                                       'words_per_sents', 'level'])
    
    col = df['documents']
    
    new_col = col.apply(lambda x: re.sub("`|’", "'", x))
    
    df['documents'] = new_col
    
    if not levels:
        col = df['level']
        new_col = []
        for level in col:
            if level == 'Ele':
                new_col.append(0)
            elif level == 'Int':
                new_col.append(1)
            else:
                new_col.append(2)
        df['level'] = new_col
     
    return df


# CEFR and Cambridge Reading Related Functions

# Path or file where the reading-level files are located
main_directory = 'Readability_dataset/'

# List of reading-level files' names matching CEFR levels from A2 to C2
level_directories = ['KET/', 'PET/', 'FCE/', 'CAE/', 'CPE/']

# List of CEFR Ratings from beginner (A2) to Advanced (C2)
cefr_ratings = ['A2', 'B1', 'B2', 'C1', 'C2']


def process_directory(cefr=True):
    """Cleans and Returns a tuple of the three lists used for data processing, analysis, and cleaning.

    Document parsing and cleaning for CEFR Reading Level Files from KET to CPE. Removes unnecessary first lines or titles,
    most subtitles, list header items, and other unnecessary lines from the files.
    Default level rating is returned in CEFR format. If set to False, will return an integer level from 0 to 5 
    for multinomial-classification.

    Parameters
    ----------
    Main Directory Path: str
    File names in matching CEFR order: list of str
    Cefr rating format: bool (Default to True)

    Returns
    -------
    (1) A list of each cleaned document as a string for data frame building
    (2) A list of of each readings level
    (3) A list of the lengths of the first line in each document
    """
    documents = []
    document_list = []
    cefr_levels = []
    first_line_lens = []
    level = 0
    
    for directory_name in level_directories:
        path = main_directory + directory_name
        directory = os.fsencode(main_directory + directory_name)
        count = 0
        first_line_delete_count = 0
        print(f'Currently processing: {path[-4:-1]}\n')
        
        for file in os.listdir(directory):
            words = []
            filename = os.fsdecode(file)
            if filename.endswith('.txt'):

                # to make a list of corresponding cefr levels as string or integer types
                if cefr:
                    cefr_levels.append(cefr_ratings[level])
                else:
                    cefr_levels.append(level)

                # to make a list of cleaned documents as a string 
                # to make a list first line lengths: 
                # Inspected above list: 45 length limit to keep first lines that are part of document with no title
                # -> most lines under this length are title lines, most over are legitmate first lines (few exceptions below)
                
                file = open(path + filename, 'r')
                
                line = file.readline()
                
                first_line_lens.append(len(line)) 
                         
                skip_first_line = True # boolean for tracking whether to skip the first line or not
                
                if len(line) > 45:
                    skip_first_line = False
                
                # First lines over 45 characters long that should be deleted: exceptions to the above rule
                first_line_deletes = ['Careless tourists', 'Build it your', 'Explore', 'BROAD']
                
                for first_words in first_line_deletes:
                    if line.startswith(first_words):
                        skip_first_line = True
                    
                if line.isupper():
                    skip_first_line = True
                
                # First lines under 45 characters long that should be kept: exceptions to the above rule
                first_line_exceptions = ['Dear', 'To:', 'TO:']
                
                for first_word in first_line_exceptions: 
                    if line.startswith(first_word):
                        skip_first_line = False
                
                if skip_first_line:
                    print('Removed First line:', line)
                    line = file.readline()
                    first_line_delete_count += 1
                
                document_string = '' # to rebuild the cleaned document
                while line != '':
                    
                    # Remove dates or address from business letters (kept one edge case with doc spacing issues)
                    if line[0].isdigit() and line.startswith(' 10 kilometers') : 
                        print('Removed:', line)
                        line = ''
                    
                    if (len(line) == 3): # Remove list header items that had space at beginning (eg. ' B.' i.e. had length 3)
                        print('Removed:', line)
                        line = ''
                    
                    if line.isupper(): # Remove any all upper case words-> usually names
                        print('Removed:', line)
                        line = ''
                    
                    # Remove edge cases and sentences which were list header items by letter
                    other_line_deletes = ['A report by', 'by', 'By Kat', 'Memo', 'Itinerary', 'Ivan Pet', 'Peter Pres', 'Stuart Har', 
                                          'Publisher ', 'transition', 'BOOK', 'Office ', 'Women on','Easter quiz', 'A Jen', 'B Mich', 
                                          'C Lisa', 'D Barb', 'E Kim', 'A.', 'B.', 'C.','D.', 'E.', 'F.', 'G.', 'H.', 'I.', 'J.']
                    
                    for start_of in other_line_deletes:
                        if line.startswith(start_of):
                            # Edge case where the complete line was deleted, so kept importance part of sentence
                            if line.startswith('E. Jane'):
                                print('Removed:', line[:3])
                                line = line[3:]
                            else:
                                print('Removed:', line)
                                line = ''
                    
                    # Reconstruct the document
                    if line:
                        current_line = line.strip()
                        document_string = " ".join((document_string, current_line))
                        # if document_string:
                        #     print(document_string)
                        #     document_list.append((current_line, level)) # keep a list of sentences and connected level
                    line = file.readline()
                file.close()
                documents.append(document_string.strip())
                # document_list.append(nltk.sent_tokenize(document_string.split()))
                count += 1
        print(f'{path[-4:-1]} has {count} files\n')
        print(f'Number of First Line Deletions: {first_line_delete_count}\n')
        level += 1
    
    return documents, cefr_levels, first_line_lens

# Original Function but not currently used in the Jupyter notebooks

def parse_directory(cefr=True):
    """Returns a tuple of the four items used for data processing, analysis, and cleaning.

    Document parsing for CEFR Reading Level Files from KET to CPE. Default level rating is returned in CEFR format.
    If set to False, will return an integer level from 0 to 5 for multinomial-classification.

    Parameters
    ----------
    Main Directory Path: str
    File names in matching CEFR order: list of str
    Cefr rating format: bool (Default to True)

    Returns
    -------
    (1) A list of lists of each reading by sentence
    (2) A list of each reading as a string
    (3) A list of of each readings level
    (4) A string of all the words in the document
    """
    documents = []
    documents_list = []
    cefr_levels = []
    words_string = ''
    level = 0
    for directory_name in level_directories:
        path = main_directory + directory_name
        directory = os.fsencode(main_directory + directory_name)
        print(f'Currently processing: {path[-4:-1]}')
        count = 0
        for file in os.listdir(directory):
            words = []
            filename = os.fsdecode(file)
            if filename.endswith('.txt'):

                # to make a single list with the whole reading as one string
                file = open(path + filename, 'r')
                
                complete_file = file.read()
                complete_file = re.sub(r'^.+?\n\n', '', complete_file)
                documents_list.append(complete_file.replace('\n', ' '))

                file.close()

                # to make a list of corresponding cefr levels
                if cefr:
                    cefr_levels.append(cefr_ratings[level])
                else:
                    cefr_levels.append(level)

                # to make a list of list of the reading broken down into individual sentences
                # to make a string of all the documents as one string
                file = open(path + filename, 'r')
                line = file.readline()
                while line != '':
                    current_line = line.strip()
                    if len(current_line) > 0:
                        words.append(current_line) 
                    words_string = " ".join((words_string, current_line))
                    line = file.readline()
                file.close()
                documents.append(words)
                count += 1
        print(f'{path[-4:-1]} has {count} files')
        level += 1
    return documents, documents_list, cefr_levels, words_string


def cefr_to_data_frame(col1, col2, col1_name='documents', col2_name='cefr_level'):
    """Returns Pandas dataframe of the CEFR reading data.

    Expects column 1 to be the reading data, and column 2 to be the level ratings for each reading.

    Parameters
    ----------
    List or Series for the first column: list or numpy array
    List or Series for the second colums: list or numpy array
    Name of first column: str (Default name is "documents")
    Name of second column: str (Default name is "cefr_level")

    Returns
    -------
    Pandas Data Frame
    """
    return pd.DataFrame({col1_name: col1, col2_name: col2})