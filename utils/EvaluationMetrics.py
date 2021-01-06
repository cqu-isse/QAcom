from six.moves import map

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
# from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from rouge import FilesRouge

def _strip(s):
    return s.strip()

def load_data(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
    lines = [l.strip() for l in lines]
    return lines

def compute_bleu(hypothesis, references, no_overlap=False, is_file=True):
    if is_file:
        with open(hypothesis, 'r') as f:
            hyp_list = f.readlines()
            f.close()
        ref_list = []
        for iidx, reference in enumerate(references):
            with open(reference, 'r') as f:
                ref_list.append(f.readlines())
                f.close()
    else:
        hyp_list = hypothesis
        ref_list = references
    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    if not no_overlap:
        scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    # print("%s: %0.6f" % (m, sc))
                    ret_scores[m] = sc
            else:
                # print("%s: %0.6f" % (method, score))
                ret_scores[method] = score
        del scorers
    return ret_scores

def compute_metrics(hypothesis, references, return_metrics=False, no_overlap=False):
    with open(hypothesis, 'r') as f:
        hyp_list = f.readlines()
        f.close()
    ref_list = []
    for iidx, reference in enumerate(references):
        with open(reference, 'r') as f:
            ref_list.append(f.readlines())
            f.close()
    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    bleu = Bleu(4)
    meteor = Meteor()
    if not no_overlap:
        scorers = [(bleu, ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
            ,(meteor, "METEOR")
            ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    # print("%s: %0.6f" % (m, sc))
                    ret_scores[m] = sc
            else:
                # print("%s: %0.6f" % (method, score))
                ret_scores[method] = score
        del scorers
        del bleu
    files_rouge = FilesRouge()
    scores = files_rouge.get_scores(hypothesis, references[0], avg=True, ignore_empty=True)
    if not return_metrics:
        print("Bleu:%.4f"%ret_scores['Bleu_4'])
        print("METEOR:%.4f" % ret_scores['METEOR'])
        print("ROUGE-1:%.4f" % scores["rouge-1"]["f"])
        print("ROUGE-2:%.4f" % scores["rouge-2"]["f"])
        print("ROUGE-L:%.4f" % scores["rouge-l"]["f"])
    del hyp_list
    del hyps
    meteor.close()
    del meteor

    if return_metrics:
        return ret_scores, scores["rouge-1"]["f"], scores["rouge-2"]["f"], scores["rouge-l"]["f"]
    else:
        return ret_scores