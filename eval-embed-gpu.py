import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress warnings
import tensorflow as tf
import sys
import math
import numpy
import random
import collections
import time
import pickle
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import itertools
import timeit

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

begin = timeit.default_timer()
inputEmbeddings = sys.argv[1]  # Embeddings in the python object text format
kb = sys.argv[2]
#kb = processPickleFile(sys.argv[2])  # Pickle database
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
    # Compute the cosine similarity between minibatch examples and all embeddings.
    normed_embeddings = tf.nn.l2_normalize(em1, dim=1)
    normed_array = tf.nn.l2_normalize(em2, dim=1)

    cosine_similarity = tf.matmul(normed_array, tf.transpose(normed_embeddings, [1,0]))

    closest_words = tf.argmax(cosine_similarity, 1)
    #norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

    init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")
        flog.write("Initialized\n")
        start = time.time()

        feed_dict = {em1: em, em2: batch}
        #l2 = session.run(closest_words, feed_dict=feed_dict)
        cosMat = session.run(cosine_similarity, feed_dict=feed_dict)

        #final_embeddings = normalized_embeddings.eval()
        end = timeit.default_timer()
        print ("Time to train model = %ds" % (end-begin) )
        flog.write ("Time to train model = %ds\n" % (end-begin) )

        data = ""
        #for i, fe in enumerate(final_embeddings):
        #    data += str(i) + "," + str(fe) + "\n"


        data += "L2s:\n"
        for cos in cosMat:
            data += str(cos) + "\n"
        outFile = inputEmbeddings + "-embeddings.out"
        with open(outFile, 'w') as fout:
            fout.write(data)

for c in cosMat:
    array = c
    sorted_array = sorted(array, reverse=True)
    print (" %s : %s" % (array, sorted_array))
