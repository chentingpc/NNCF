import numpy as np
from sampler import MultinomialSampler

def get_sampler(ratings, neg_dist='unigram', neg_sampling_power=0.75, rand_seed=0, batch_mode=True):
    '''
    Return a sampling function

    Assuming the triplets are in shape of (user, item, rating), i.e. dst/item will be sampled as
        negatives
    '''
    assert neg_dist == 'uniform' or neg_dist == 'unigram', [neg_dist]

    dist = np.array([0.] * (np.max(ratings[:, [1]])+1))
    for src, dst, _ in ratings:
        dist[dst] += 1
    if neg_dist == 'uniform':
        dist[dist > 0] = 1
        # dist[:] = 1
    s = MultinomialSampler(dist, dist.size, neg_sampling_power, rand_seed)
    if batch_mode:
        return s.sample_batch
    else:
        return s.sample

data = np.vstack([range(1000), range(1000), range(1000)]).T

sample = get_sampler(data, neg_dist='uniform', batch_mode=False)
sample_batch = get_sampler(data, neg_dist='uniform', batch_mode=True)

print sample()
print sample_batch(10)
