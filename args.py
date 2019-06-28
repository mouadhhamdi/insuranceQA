import argparse

def get_setup_args():
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_path',
                        type=str,
                        default='/home/mouadh/Desktop/insuranceQA/V2/vocabulary',
                        help='path to vocabulary')

    parser.add_argument('--label_path',
                        type=str,
                        default= '/home/mouadh/Desktop/insuranceQA/V2/InsuranceQA.label2answer.token.encoded',
                        help='path to unzipped label2answer file')

    parser.add_argument('--train_path',
                        type=str,
                        default='/home/mouadh/Desktop/insuranceQA/V2/InsuranceQA.question.anslabel.token.500.pool.solr.train.encoded',
                        help='path to unzipped anslabel train file')

    parser.add_argument('--test_path',
                        type=str,
                        default= '/home/mouadh/Desktop/insuranceQA/V2/InsuranceQA.question.anslabel.token.500.pool.solr.test.encoded',
                        help='path to unzipped anslabel test file')

    parser.add_argument('--glove_path',
                        type=str,
                        default='/home/mouadh/Desktop/insuranceQA/glove.840B.300d/glove.840B.300d.txt',
                        help='path to glove word embedding')


    parser.add_argument('--glove_dim',
                        type=int,
                        default=300,
                        help='Size of GloVe word vectors to use')

    parser.add_argument('--glove_num_vecs',
                        type=int,
                        default=2196017,
                        help='Number of GloVe vectors')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help='Number of features in encoder hidden layers.')
    
    parser.add_argument('--word_emb_file',
                        type=str,
                        default= '/home/mouadh/Desktop/insuranceQA/glove.840B.300d/word_embedding',
                        help='Path to the word embedding file')
    
    parser.add_argument('--word2idx_file',
                        type=str,
                        default= '/home/mouadh/Desktop/insuranceQA/glove.840B.300d/word2idx',
                        help='Path to the word to index file')
    
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed')
    
    parser.add_argument('--create_matrix_embedding',
                        type=bool,
                        default=True,
                        help='create glove matrix embedding')
    
    parser.add_argument('--lr',
                        type=float,
                        default=0.5,
                        help='Learning rate.')
    
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')

    parser.add_argument('--margin', 
                        type=float, 
                        default=0.2, 
                        help='margin for loss function')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128, 
                        help='input batch size')
    
    parser.add_argument('--use_glove', 
                        type=bool, 
                        default=False, 
                        help='use glove embeddings')
    
    parser.add_argument('--embd_size', 
                        type=int, 
                        default=200, 
                        help='embedding size')
    

    
    args = parser.parse_args(args=[])

    return args