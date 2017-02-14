import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress warnings
import tensorflow as tf
import sys
import math
import numpy
import random
import collections
import time
import operator
from collections import defaultdict as ddict
import pickle
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import itertools
import timeit
import time
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('EVAL-EMBED')
evalMethod = "cosine"

def processFile(datafile):
    with open(datafile,'r')as fin:
        data = fin.read()

    records = data.split(']')
    # Remove the last element (extra newline)
    del(records[-1])
    embeddings = [[] for _ in range(len(records))]
    for i,r in enumerate(records):
        embeddings_str = r.split(',[')[1].split()
        for e in embeddings_str:
            embeddings[i].append(float(e))

    return numpy.array(embeddings)

def processPickleFile(datafile):
    with open(datafile, 'rb') as fin:
        data = pickle.load(fin)
    return data

if len(sys.argv) < 4:
    print ("Usage: python %s <embeddings.txt> <kb.bin> <TOPK>" % (sys.arg[0]))
    sys.exit()
begin = timeit.default_timer()
inputEmbeddings = sys.argv[1]  # Embeddings in the python object text format
kb = sys.argv[2]
TOPK = int(sys.argv[3])
logfile = inputEmbeddings + ".log"
flog = open(logfile, 'w')

graph = tf.Graph()
with graph.as_default():
    with tf.device('/gpu:2'):
        # Init embeddings
        em = processFile(inputEmbeddings)
        N = len(em)
        dim = len(em[0])
        kbRecords = processPickleFile(kb)

        # Prepare batch: embeddings of all heads from the test dataset
        batch = []
        test = kbRecords['test_subs']
        for t in test:
            head = t[0]
            batch.append(em[head])
        M = len(test)
        dimBatch = len(em[0])

    em1 = tf.placeholder(tf.float32, [N, dim])
    em2 = tf.placeholder(tf.float32, [M, dimBatch])
    # Compute the cosine similarity between test batch and all embeddings.
    normed_embeddings = tf.nn.l2_normalize(em1, dim=1)
    normed_array = tf.nn.l2_normalize(em2, dim=1)

    cosine_similarity = tf.matmul(normed_array, tf.transpose(normed_embeddings, [1,0]))

    #closest_words = tf.argmax(cosine_similarity, 1)
    #norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

    init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        init.run()
        log.info("Initialized\n")
        flog.write("Initialized\n")
        start = time.time()

        feed_dict = {em1: em, em2: batch}
        cosMat = session.run(cosine_similarity, feed_dict=feed_dict)

        end = timeit.default_timer()
        print ("Time to train model = %ds" % (end-begin) )
        flog.write ("Time to train model = %ds\n" % (end-begin) )

        #data = ""
        #data += "L2s:\n"
        #for cos in cosMat:
        #    data += str(cos) + "\n"
        #outFile = inputEmbeddings + "-embeddings.out"
        #with open(outFile, 'w') as fout:
        #    fout.write(data)

# Num Rows in cosine matrix must be equal to number of test triples
if (len(cosMat) != len(test)):
    print ("FATAL PROBLEM")
    sys.exit()

log.info("Length check passed %d" % (len(cosMat)))
flog.write("Length check passed %d" % (len(cosMat)))
out = []
for i, triple in enumerate(test):
    log.info("Tuple(%d) - (%d, %d, %d) : " % (i, triple[0], triple[1], triple[2]))
    flog.write("Tuple(%d) - (%d, %d, %d) : " % (i, triple[0], triple[1], triple[2]))
    array = cosMat[i]
    cos_dict = ddict()
    for j,a in enumerate(array):
        cos_dict[j] = a
    head = triple[0]
    tail = triple[1]
    relation = triple[2]
    sorted_dict = sorted(cos_dict.items(), key = operator.itemgetter(1), reverse=True)

    log.info("%d cosine results sorted, " % (len(cos_dict)))
    flog.write("%d cosine results sorted, " % (len(cos_dict)))
    found = False
    for k,v in enumerate(sorted_dict):
        if k == TOPK:
            break
        if v[0] == tail:
            out.append((head, tail, k))
            found = True
            break
    if k == TOPK:
        out.append((head, tail, -1))
    log.info("Position found : %d\n" % (k))
    flog.write("Position found : %d\n" % (k))

outFile = sys.argv[1] + "-" + "TOP-" + str(TOPK) + "-" + evalMethod + ".eval.out"
data = "{"
for i, pairs in enumerate(out):
    data += str(i) + ": {"
    for p in pairs:
        data += str(p) + " "
    data += "}"
    data += "\n"
data += "}"
with open(outFile, 'w') as fout:
    fout.write(data)
