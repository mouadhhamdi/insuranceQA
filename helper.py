# Imports
import gzip

# Path to the files
PATH = '/home/mouadh/Desktop/AXA/insuranceQA/V2/'

# Read the vocabulary 
def construct_vocab(file):
    '''
    The vocabulary will be a dictionary with the word as value
    and the idx as a key
    '''
    vocabulary = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split("\t")
            # Remove the return to the line
            vocabulary[(key)] = val.replace('\n','')
    return vocabulary
# Construct vocabulary
print("Vocabulary constructed")
vocabulary = construct_vocab(PATH + "vocabulary")

# This function will open a gzip files
def read_data(file, name):
    with gzip.open(file,'rb') as f:        
        lines = [x.decode('utf8').strip().split('\t') for x in f.readlines()]
    if name == "label2answer":
        print("Reading label2answer data. \nThis data format is: <Answer Label><TAB><Answer Text> \n")
        lines = [[int(l[0]), l[1].split(' ')] for l in lines]
    elif name == "question.anslabel":
        print("Reading questions.anslabel data.\nThis data foramt is: <Domain><TAB><QUESTION><TAB><Groundtruth>\n")
        lines = [ [l[0], l[1].split(' '), l[2].split(' ') ] for l in lines]
    else:
        print("Reading Train/Test/Validation file.\nThis data format is: <Domain><TAB><QUESTION><TAB><Groundtruth><TAB><Pool>\n")
        lines = [[l[0], l[1].split(' '), l[2].split(' '), l[3].split(' ')] for l in lines]
    return lines

# This function will convert a list of indexes to list of words
def convert_from_idx_str(list):
    list = [vocabulary[l] for l in list]
    return list
