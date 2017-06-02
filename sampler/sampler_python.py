'''
This file mainly contains Sampler class, which helps sample from discrete dist.
'''

import random
import bisect
import numpy as np


class weighted_choice_b(object):
    '''
    Sampling from categorical distribution
    '''
    def __init__(self, weights, seed=4224):
        '''
        weights: list of floating numbers
        '''
        totals = []
        running_total = 0
        random.seed(seed)

        for w in weights:
            running_total += w
            totals.append(running_total)
        self.totals = totals
        self.running_total = running_total

    def sample(self):
        rnd = random.random() * self.running_total
        return bisect.bisect_right(self.totals, rnd)

    def sample_gen(self, size=1):
        # this return a generator, doesn't make too much difference to sample size times
        for i in range(size):
            rnd = random.random() * self.running_total
            yield bisect.bisect_right(self.totals, rnd)


class Sampler(object):
    def __init__(self, node_count, sampling_method='uniform'):
        if sampling_method == 'uniform':
            nodes = list(node_count.keys())
            num_node = len(nodes)
            self.sampling_method = 0
            self.nodes = nodes
            self.num_node = num_node
            self.uniform_prob = 1./num_node
            print '[!INFO] Sampler initialized in uniform. Total nodes %d.' % num_node
        else:
            index2node = []
            node2index = dict()
            weights = []
            weights_sum = 0
            for i, node in enumerate(node_count):
                index2node.append(node)
                node2index[node] = i
                weight = node_count[node]
                weights.append(weight)
                weights_sum += weight
            for i in range(len(weights)):
                weights[i] /= float(weights_sum)
            wb = weighted_choice_b(weights)
            self.sampling_method = 1
            self.wb = wb
            self.probs = np.array(weights)
            self.index2node = np.array(index2node)
            self.node2index = node2index
            assert len(node2index) == len(index2node)
            print '[!INFO] Sampler initialized in unigram. Total nodes %d, total counts %d.' \
                % (len(node2index), weights_sum)

        assert self.sampling_method == 0 or self.sampling_method == 1

    def sample(self):
        if self.sampling_method == 0:
            return random.choice(self.nodes)
        else:
            return self.index2node[self.wb.sample()]

    def sample_wprob(self):
        if self.sampling_method == 0:
            return [random.choice(self.nodes), self.uniform_prob]
        else:
            index = self.wb.sample()
            s = self.index2node[index]
            p = self.probs[index]
            return [s, p]

    def get_prob_anode(self, node):
        if self.sampling_method == 0:
            return self.uniform_prob
        else:
            return self.probs[self.node2index[node]]

if __name__ == '__main__':
    # simple test
    test_count = {0: 5, 1: 3, 2: 2}
    sampler = Sampler(test_count, 'unigram')
    counter = [0, 0, 0]
    for i in range(10000):
        a = sampler.sample_wprob()[0]
        counter[a] += 1
    print counter
