from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
import re

dataset = "blogcatalog"
#dataset = "flickr"
#dataset = "youtube"
methods = {"group_sample": "Stratified",
           "group_neg_shared": "Stratified with N.S.",
           "neg_shared": "Negative Sharing",
           "original": "IID"}

results_all = {}
for method in methods:
    results = {'stamp': [],
               'epoch': [],
               'cost': [],
               'micro': [],
               'macro': [],
               'mm': []
              }
    result_folder = "../network-{}/mf/{}/".format(dataset, method)
    filenames = os.listdir(result_folder)
    log_file_main = None
    log_file_class = None
    for filename in filenames:
        if filename == "log-classification":
            assert log_file_class is None, log_file_class
            log_file_class = result_folder + filename
        elif filename.startswith("log"):
            assert log_file_main is None, log_file_main
            log_file_main = result_folder + filename
    assert log_file_main is not None and log_file_class is not None, [method, filenames]
    with open(log_file_main) as fp:
        text = fp.read()
        _result_main = re.findall("stamp ([\d\.]+) epoch (\d+) .* cost ([\-\.\d]+)", text)
    with open(log_file_class) as fp:
        text = fp.read()
        _result_class = re.findall("'micro': ([\.\d]+).* 'macro': ([\.\d]+)", text)
        if len(_result_class) - len(_result_main) == 1:
            _result_class = _result_class[1:]
    assert len(_result_main) == len(_result_class), [len(_result_main), len(_result_class)]
    for part1, part2 in zip(_result_main, _result_class):
        stamp, epoch, cost = map(float, part1)
        micro, macro = map(float, part2)
        epoch = int(epoch)
        results['stamp'].append(stamp)
        results['epoch'].append(epoch)
        results['cost'].append(cost)
        results['micro'].append(micro)
        results['macro'].append(macro)
        results['mm'].append((macro + micro) / 2.)
    results_all[method] = results

# Plot epoch vs cost.
for method in methods:
    results = results_all[method]
    plt.plot(results['epoch'][1:], results['cost'][1:], label=methods[method])
plt.title(dataset)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot epoch vs macro.
for method in methods:
    results = results_all[method]
    plt.plot(results['epoch'], results['macro'], label=methods[method])
plt.title(dataset)
plt.xlabel("Epoch")
plt.ylabel("Macro")
plt.legend(loc="lower right")
plt.show()

# Plot epoch vs micro.
for method in methods:
    results = results_all[method]
    plt.plot(results['epoch'], results['micro'], label=methods[method])
plt.title(dataset)
plt.xlabel("Epoch")
plt.ylabel("Micro")
plt.legend(loc="lower right")
plt.show()

# Plot epoch vs micro + macro.
for method in methods:
    results = results_all[method]
    plt.plot(results['epoch'], results['mm'], label=methods[method])
plt.title(dataset)
plt.xlabel("Epoch")
plt.ylabel("(Macro + Micro) / 2")
plt.legend(loc="lower right")
plt.savefig("{}_epoch_vs_perf.pdf".format(dataset), bbox_inches='tight')
plt.show()

# Plot time vs micro + macro.
for method in methods:
    results = results_all[method]
    plt.plot(results['stamp'], results['mm'], label=methods[method])
plt.title(dataset)
plt.xlabel("Wall time")
plt.ylabel("(Macro + Micro) / 2")
plt.legend(loc="lower right")
plt.savefig("{}_wtime_vs_perf.pdf".format(dataset), bbox_inches='tight')
plt.show()

