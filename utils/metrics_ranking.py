
import numpy as np
import bottleneck


def eval_multiple(true_scores, pred_scores, topk):
    idx = bottleneck.argpartition(-pred_scores, topk)[:topk]
    noise = np.random.random(topk)
    if not isinstance(pred_scores, np.ndarray):
        pred_scores = np.array(pred_scores)
    if not isinstance(true_scores, np.ndarray):
        true_scores = np.array(true_scores)
    rec = sorted(zip(pred_scores[idx], noise, true_scores[idx]), reverse=True)
    nhits = 0.
    nhits_topk = 0.
    k = topk if topk >= 0 else len(rec)
    sumap = 0.0
    for i in range(len(rec)):
        if rec[i][-1] != 0.:
            nhits += 1.0
            if i < k:
                nhits_topk += 1
                sumap += nhits / (i+1.0)
    nhits = np.sum(true_scores)
    if nhits != 0:
        sumap /= min(nhits, k)
        map_at_k = sumap
        recall_at_k = nhits_topk / nhits
        precision_at_k = nhits_topk / k
    else:
        map_at_k = 0.
        recall_at_k = 0.
        precision_at_k = 0.

    return map_at_k, recall_at_k, precision_at_k


def eval_multiple_original(true_scores, pred_scores, topk):
    noise = np.random.random(len(true_scores))
    rec = sorted(zip(pred_scores, noise, true_scores), reverse=True)
    nhits = 0.
    nhits_topk = 0.
    k = topk if topk >= 0 else len(rec)
    sumap = 0.0
    for i in range(len(rec)):
        if (rec[i][-1] != 0.):
            nhits += 1.0
            if i < k:
                nhits_topk += 1
                sumap += nhits / (i+1.0)
    if nhits != 0:
        sumap /= min(nhits, k)
        map_at_k = sumap
        recall_at_k = nhits_topk / nhits
        precision_at_k = nhits_topk / k
    else:
        map_at_k = 0.
        recall_at_k = 0.
        precision_at_k = 0.

    return map_at_k, recall_at_k, precision_at_k


def eval_apk(true_scores, pred_scores, topk):
    idx = bottleneck.argpartition(-pred_scores, topk)[:topk]  # find the top-k smallest
    noise = np.random.random(topk)
    if not isinstance(pred_scores, np.ndarray):
        pred_scores = np.array(pred_scores)
    if not isinstance(true_scores, np.ndarray):
        true_scores = np.array(true_scores)
    rec = sorted(zip(pred_scores[idx], noise, true_scores[idx]), reverse=True)
    nhits = 0.
    k = topk if topk >= 0 else len(rec)
    sumap = 0.0
    for i in range(len(rec)):
        if (rec[i][-1] != 0.):
            nhits += 1.0
            if i < k:
                sumap += nhits / (i+1.0)
            else:
                break
    nhits = np.sum(true_scores)
    if nhits != 0:
        sumap /= min(nhits, k)
        return sumap
    else:
        return 0.


def eval_apk_original(true_scores, pred_scores, topk):
    noise = np.random.random(len(true_scores))
    rec = sorted(zip(pred_scores, noise, true_scores), reverse=True)
    nhits = 0.
    k = topk if topk >= 0 else len(rec)
    sumap = 0.0
    for i in range(len(rec)):
        if (rec[i][-1] != 0.):
            nhits += 1.0
            if i < k:
                sumap += nhits / (i+1.0)
    if nhits != 0:
        sumap /= min(nhits, k)
        return sumap
    else:
        return 0.


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if len(actual):
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
