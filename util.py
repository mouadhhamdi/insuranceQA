import torch
from tqdm import tqdm
import numpy as np
import ujson as json
from torch.autograd import Variable

PAD = '<PAD>'


def load_vocabulary(vocab_path, label_path):
    id_to_word = {}
    with open(vocab_path) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            if d[0] not in id_to_word:
                id_to_word[d[0]] = d[1]

    label_to_ans = {}
    label_to_ans_text = {}
    with open(label_path) as f:
        lines = f.readlines()
        for l in lines:
            label, answer = l.rstrip().split('\t')
            if label not in label_to_ans:
                label_to_ans[label] = answer
                label_to_ans_text[label] = [id_to_word[t] for t in answer.split(' ')]
    return id_to_word, label_to_ans, label_to_ans_text


def load_data_train(fpath, id_to_word, label_to_ans_text):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')] # question
            poss = [label_to_ans_text[t] for t in d[2].split(' ')] # ground-truth
            negs = [label_to_ans_text[t] for t in d[3].split(' ') if t not in d[2]] # candidate-pool without ground-truth
            for pos in poss:
                data.append((q, pos, negs))
    return data

def load_data_test(fpath, id_to_word, label_to_ans_text):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for l in lines[12:]:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')] # question
            poss = [t for t in d[2].split(' ')] # ground-truth
            cands = [t for t in d[3].split(' ')] # candidate-pool
            data.append((q, poss, cands))
    return data


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def padding(data, max_sent_len, pad_token):
    pad_len = max(0, max_sent_len - len(data))
    data += [pad_token] * pad_len
    data = data[:max_sent_len]
    return data


def make_vector(data, w2i, seq_len):
    ret_data = [padding([w2i[w] for w in d], seq_len, w2i[PAD]) for d in data]
    return to_var(torch.LongTensor(ret_data))



def get_embedding(data_type, emb_file=None, vec_size=None, num_vectors=None):
    print("Pre-processing {} vectors...".format(data_type))
    embedding_dict = {}
    assert vec_size is not None
    with open(emb_file, "r", encoding="utf-8") as fh:

        for line in tqdm(fh, total=num_vectors):
            array = line.split()
            word = "".join(array[0:-vec_size])

            vector = list(map(float, array[-vec_size:]))
            embedding_dict[word] = vector
    print("{}  tokens have corresponding {} embedding vector".format(
        len(embedding_dict), data_type))

        
    PAD = "<PAD>"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 1)}
    token2idx_dict[PAD] = 0
    embedding_dict[PAD] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)
            

def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.
    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.
    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor

def loss_fn(pos_sim, neg_sim, margin):
    loss = margin - pos_sim + neg_sim
    if loss.data[0] < 0:
        loss.data[0] = 0
    return loss



def get_available_devices():
    """Get IDs of all available GPUs.
    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


