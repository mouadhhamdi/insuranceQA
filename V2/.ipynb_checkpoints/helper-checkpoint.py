# Imports
import gzip
import os


# Path to the files
PATH = '/home/mouadh/Desktop/AXA/insuranceQA/insuranceQA/V2/'

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

# This function will open a gzip files
def read_data(file, name):
    with gzip.open(file,'rb') as f:        
        lines = [x.decode('utf8').strip().split('\t') for x in f.readlines()]
    if name == "label2answer":
        lines = [[int(l[0]), l[1].split(' ')] for l in lines]
    elif name == "question.anslabel":
        lines = [ [l[0], l[1].split(' '), l[2].split(' ') ] for l in lines]
    else:
        # <Domain><TAB><QUESTION><TAB><Groundtruth><TAB><Pool>
        lines = [[l[0], l[1].split(' '), l[2].split(' '), l[3].split(' ')] for l in lines]
    return lines

# This function will convert a list of indexes to list of words
def convert_from_idx_str(list):
    vocabulary = construct()
    list = [vocabulary[l] for l in list]
    return list
