import numpy as np
import os
import pickle
import sys
from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from utils.EvaluationMetrics import compute_metrics
from utils.EvaluationMetrics import compute_bleu
from utils.IgnoreRate import CalculateIgnoreRate
from utils.VocabularyAndMapping import BuildMappings
from utils.DifferentialEvolution import Population

# TF mode = 1  /  TF-idf mode = 2
Word_bag_mode = 'TF-IDF'

# Strategy = 1/2/3
Strategy = 3

calculation_refs = 0
global_bleu = 0

class obj:
    def __init__(self, key=0, weight=0.0):
        self.key = key
        self.weight = weight

class Calculation_Refs:
    def __init__(self, ignore_rate, mapping, test_diff_score, diff_list, msg_list, test_msgs, gen_msgs):
        self.ignore_rate = ignore_rate
        self.mapping = mapping
        self.test_diff_score = test_diff_score
        self.diff_list = diff_list
        self.msg_list = msg_list
        self.test_msgs = test_msgs
        self.gen_msgs = gen_msgs

# load a numpy.array from a .npy file
def load_npy(npypath):
    return np.load(npypath)

# load lines from a file
def load_data(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
    lines = [l.strip() for l in lines]
    return lines

def load_vocabulary(dataset, type):
    Vcb_path = '.\\vocabulary\\'+dataset+'.'+type+'.pkl'
    pkl = open(Vcb_path, 'rb')
    vocabulary = pickle.load(pkl)
    return vocabulary

def IsCandidate(sentence1, sentence2):
    a = [x for x in sentence1 if x in sentence2]
    if len(a) < 1:
        return 0
    else:
        return 1

def nonzero2list(nonzero):
    k = 0
    lis = []
    gather = []
    p = -1
    for i in nonzero[0]:
        p = p + 1
        if k == i:
            lis.append(nonzero[1][p])
        else:
            gather.append(lis)
            while k < i - 1:
                k = k + 1
                lis = []
                gather.append(lis)
            lis = []
            k = i
            lis.append(nonzero[1][p])
    gather.append(lis)
    return gather

def similarity(train_diffs, test_diffs , n=5):
    counter = CountVectorizer()
    global Word_bag_mode
    if Word_bag_mode == 'TF':
        # TF mode
        train_matrix = counter.fit_transform(train_diffs)
        test_matrix = counter.transform(test_diffs)
    elif Word_bag_mode == 'TF-IDF':
        # TF-IDF mode
        transformer = TfidfTransformer()
        train_matrix = transformer.fit_transform(counter.fit_transform(train_diffs))
        test_matrix = transformer.transform(counter.transform(test_diffs))

    similarities = cosine_similarity(test_matrix, train_matrix)
    test_diff_score = []
    count = -1
    for idx, test_simi in enumerate(similarities):
        count += 1
        score = 0
        global Strategy
        if Strategy == 1:
            str1 = train_diffs[test_simi.argsort()[-1]]
            str2 = test_diffs[idx]
            score = obj(count, cosine_similarity(counter.transform([str1]), counter.transform([str2]))[0][0])
        elif Strategy == 2:
            score_temp = 0
            for i in range(n):
                str1 = train_diffs[test_simi.argsort()[-(i+1)]]
                str2 = test_diffs[idx]
                score_temp += cosine_similarity(counter.transform([str1]), counter.transform([str2]))[0][0] / (i+1)
            score = obj(count, score_temp)
        elif Strategy == 3:
            ref = []
            str2 = test_diffs[idx]
            for i in range(n):
                str1 = train_diffs[test_simi.argsort()[-(i + 1)]]
                ref.append(str1.split())
            smooth = SmoothingFunction()
            score = obj(count, sentence_bleu(ref, str2.split(), weights=(0.25, 0.25, 0.25, 0.25),
                                             smoothing_function=smooth.method1))
        test_diff_score.append(score)
    return test_diff_score

def Generation_File(alpha, beta, gamma, relevance, output_path):
    global calculation_refs
    ref_output_path = output_path + ".ref"
    fw_ref = open(ref_output_path, "w")
    fw_output = open(output_path, "w")
    count = -1
    output_num = 0
    for diff in calculation_refs.diff_list:
        count += 1
        msg = calculation_refs.msg_list[count]
        total_num = len(diff)
        trans_num = 0
        msg_words = []
        for word in diff:
            if calculation_refs.ignore_rate[word] < alpha:
                trans_num += IsCandidate(calculation_refs.mapping[word], msg)
            else:
                total_num -= 1
            for msg_word in calculation_refs.mapping[word]:
                msg_words.append(msg_word)
        correct_num = 0
        for word in msg:
            if word in msg_words:
                correct_num += 1
        if total_num > 0:
            recall = float(trans_num) / float(total_num)
            if (len(msg)) == 0:
                precision = 0
            else:
                precision = float(correct_num) / float(len(msg))
            if (recall >= beta) and (precision >= gamma) and (
                calculation_refs.test_diff_score[count].weight >= relevance) and len(diff) > 0:
                fw_ref.write(calculation_refs.test_msgs[count] + '\n')
                fw_output.write(calculation_refs.gen_msgs[count] + '\n')
                output_num += 1
    print("Preserved-Ratio:%0.4f" % (int(output_num)/len(calculation_refs.diff_list)))
    fw_ref.close()
    fw_output.close()
    print("Done. Evaluate the metrics...")
    # print(compute_bleu(output_path, [ref_output_path])['Bleu_4'])
    compute_metrics(output_path, [ref_output_path])

def Generation(alpha, beta, gamma, relevance):
    global calculation_refs
    ref = []
    output = []
    count = -1
    output_num = 0
    for diff in calculation_refs.diff_list:
        count += 1
        msg = calculation_refs.msg_list[count]
        total_num = len(diff)
        untrans_num = 0
        msg_words = []
        for word in diff:
            if calculation_refs.ignore_rate[word] < alpha:
                untrans_num += IsCandidate(calculation_refs.mapping[word], msg)
            else:
                total_num -= 1
            for msg_word in calculation_refs.mapping[word]:
                msg_words.append(msg_word)
        correct_num = 0
        for word in msg:
            if word in msg_words:
                correct_num += 1
        if total_num > 0:
            untrans_rate = untrans_num / total_num
            if(len(msg)) == 0:
                precision = 0
            else:
                precision = float(correct_num) / float(len(msg))
            if (untrans_rate >= beta) and (precision >= gamma) and (calculation_refs.test_diff_score[count].weight >= relevance):
                ref.append(calculation_refs.test_msgs[count])
                output.append(calculation_refs.gen_msgs[count])
                output_num +=1
    Bleu = compute_bleu(output, [ref], is_file=False)['Bleu_4']
    return output_num, Bleu

def obj_fuc(v):
    global global_bleu
    output_num, Bleu = Generation(v[0], v[1], v[2], v[3])
    if Bleu < 0.4:
        return 100
    else:
        global_bleu = Bleu
        return -output_num

def QAcom_train(dataset, approach):
    print("Dataset: %s  Approach: %s" % (dataset, approach))
    train_diff_path = ".\\data\\" + dataset + "\\" + dataset + ".train.diff"
    valid_diff_path = ".\\data\\" + dataset + "\\" + dataset + ".valid.diff"
    valid_msg_path = ".\\data\\" + dataset + "\\" + dataset + ".valid.msg"
    original_path = ".\\data\\" + dataset + "\\" + approach + "." + dataset + ".valid.gen.msg"

    train_diffs = load_data(train_diff_path)
    valid_diffs = load_data(valid_diff_path)
    valid_msgs = load_data(valid_msg_path)
    original_gen_msgs = load_data(original_path)

    valid_diff_score = similarity(train_diffs, valid_diffs)

    Mapping_path = ".\\Mapping\\" + dataset + ".npy"
    if not os.path.exists(Mapping_path):
        BuildMappings(dataset)
    Mapping = load_npy(Mapping_path)

    IgnoreRate_path = ".\\IgnoreRate\\" + dataset + ".npy"
    if not os.path.exists(IgnoreRate_path):
        CalculateIgnoreRate(dataset)
    IgnoreRate = load_npy(IgnoreRate_path)

    diff_vocabulary = load_vocabulary(dataset, 'diff')
    msg_vocabulary = load_vocabulary(dataset, 'msg')

    counter1 = CountVectorizer(lowercase=True, vocabulary=diff_vocabulary)
    diff_matrix = counter1.fit_transform(valid_diffs)
    counter2 = CountVectorizer(lowercase=True, vocabulary=msg_vocabulary)
    msg_matrix = counter2.fit_transform(original_gen_msgs)

    diff_nonzero = sparse.csr_matrix(diff_matrix).nonzero()
    msg_nonzero = sparse.csr_matrix(msg_matrix).nonzero()
    diff_list = nonzero2list(diff_nonzero)
    msg_list = nonzero2list(msg_nonzero)

    global calculation_refs
    calculation_refs = Calculation_Refs(IgnoreRate, Mapping, valid_diff_score, diff_list, msg_list, valid_msgs, original_gen_msgs)

    de = Population(min_range=0, max_range=1, dim=4, factor=0.5, rounds=10, size=5, object_func=obj_fuc)
    num, thresholds = de.evolution()
    print("Val Bleu: %0.4f  Val Preserved_Ratio: %0.4f"%(global_bleu, num/len(valid_msgs)))
    print("Ignore-rate: %0.3f   P_m: %0.3f  R_m: %0.3f  Relevance: %0.3f"%(thresholds[0], thresholds[1], thresholds[2], thresholds[3]))
    QAcom(dataset, approach, thresholds[0], thresholds[1], thresholds[2], thresholds[3])

def QAcom(dataset, approach, alpha, beta, gamma, relevance):
    # print("Dataset: %s  Approach: %s" % (dataset, approach))
    train_diff_path = ".\\data\\" + dataset + "\\" + dataset + ".train.diff"
    test_diff_path = ".\\data\\" + dataset + "\\" + dataset + ".test.diff"
    test_msg_path = ".\\data\\" + dataset + "\\" + dataset + ".test.msg"
    original_path = ".\\data\\" + dataset + "\\" + approach + "." + dataset + ".gen.msg"
    QAcom_output_path = ".\\archive\\" + approach + "." + dataset + ".QAcom.filtered"

    train_diffs = load_data(train_diff_path)
    test_diffs = load_data(test_diff_path)
    original_gen_msgs = load_data(original_path)

    # The ground truth is used to generate the reference message file (used to calculate the evaluation metrics).
    test_msgs = load_data(test_msg_path)

    Mapping_path = ".\\Mapping\\" + dataset + ".npy"
    if not os.path.exists(Mapping_path):
        BuildMappings(dataset)
    Mapping = load_npy(Mapping_path)

    IgnoreRate_path = ".\\IgnoreRate\\" + dataset + ".npy"
    if not os.path.exists(IgnoreRate_path):
        CalculateIgnoreRate(dataset)
    IgnoreRate = load_npy(IgnoreRate_path)

    diff_vocabulary = load_vocabulary(dataset, 'diff')
    msg_vocabulary = load_vocabulary(dataset, 'msg')

    counter1 = CountVectorizer(lowercase=True, vocabulary=diff_vocabulary)
    diff_matrix = counter1.fit_transform(test_diffs)
    counter2 = CountVectorizer(lowercase=True, vocabulary=msg_vocabulary)
    msg_matrix = counter2.fit_transform(original_gen_msgs)

    diff_nonzero = sparse.csr_matrix(diff_matrix).nonzero()
    msg_nonzero = sparse.csr_matrix(msg_matrix).nonzero()
    diff_list = nonzero2list(diff_nonzero)
    msg_list = nonzero2list(msg_nonzero)

    test_diff_score = similarity(train_diffs, test_diffs)

    global calculation_refs
    calculation_refs = Calculation_Refs(IgnoreRate, Mapping, test_diff_score, diff_list, msg_list, test_msgs, original_gen_msgs)
    Generation_File(alpha, beta, gamma, relevance, QAcom_output_path)

if __name__ == '__main__':
    dataset = sys.argv[1]
    approach = sys.argv[2]
    QAcom_train(dataset, approach)