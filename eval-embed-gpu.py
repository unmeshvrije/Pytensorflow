import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress warnings
import tensorflow as tf
import sys
import math
import numpy
import random
import collections
import argparse
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
from numpy import argsort

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('EVAL-EMBED')

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

parser = argparse.ArgumentParser(prog = "Evaluate  embeddings based on L1/cosine distances")
parser.add_argument('--fin', type=str, help = "Embeddings in the python object text format")
parser.add_argument('--fdb', type=str, help = "Pickle database")
parser.add_argument('--topk', type=int, help = "TOPK value for evaluation")
parser.add_argument('--eval-method', type=str, help = "Evaluation method", default='cosine')
parser.add_argument('--dev', type=str, help = "Whether to run on CPU or GPU", default='gpu')
parser.add_argument('--filter', action='store_const',const=True, default=False, help = "Whether to use filter or not")

args = parser.parse_args()
begin = timeit.default_timer()
inputEmbeddings = args.fin # Embeddings in the python object text format
kb = args.fdb
TOPK = int(args.topk)
evalMethod = args.eval_method
logfile = inputEmbeddings + ".log"
flog = open(logfile, 'w')
dev = args.dev
filter = args.filter

def num_related_heads(relation, graph):
    all_entities_as_head = list(graph['relations_head'][relation].keys())
    return all_entities_as_head

def num_related_tails(relation, graph):
    all_entities_as_tail = list(graph['relations_tail'][relation].keys())
    return all_entities_as_tail

def make_graph(triples, N, M):
    graph_outgoing = [ddict(list) for _ in range(N)]
    graph_incoming = [ddict(list) for _ in range(N)]
    graph_relations_head = [ddict(list)for _ in range(M)]
    graph_relations_tail = [ddict(list)for _ in range(M)]
    for t in triples:
        head = t[0]
        tail = t[1]
        relation = t[2]
        graph_outgoing[head][relation].append(tail)
        graph_incoming[tail][relation].append(head)
        graph_relations_head[relation][head].append(tail)
        graph_relations_tail[relation][tail].append(head)

    return {'outgoing': graph_outgoing, 'incoming': graph_incoming, 'relations_head': graph_relations_head, 'relations_tail':graph_relations_tail}

graph = tf.Graph()
with graph.as_default():
    with tf.device('/'+dev+':2'):
        # Init embeddings
        em = processFile(inputEmbeddings)
        N = len(em)
        dim = len(em[0])
        kbRecords = processPickleFile(kb)

        # Prepare batch: embeddings of all heads from the test dataset
        batchHeads = []
        batchTails = []
        if (N != len(kbRecords['entities'])):
            log.info("FATAL problem")
            sys.exit()
        M = len(kbRecords['relations'])
        train = kbRecords['train_subs']
        kbGraph = make_graph(train, N, M)
        test = kbRecords['test_subs']
        for t in test:
            head = t[0]
            tail = t[1]
            batchHeads.append(em[head])
            batchTails.append(em[tail])
        cntTestTriples = len(test)
        dimBatch = len(em[0])

    if evalMethod == "cosine":
        allEmbeddings = tf.placeholder(tf.float32, [N, dim])
        em2 = tf.placeholder(tf.float32, [cntTestTriples, dimBatch])
        # Compute the cosine similarity between test batch and all embeddings.
        normed_embeddings = tf.nn.l2_normalize(allEmbeddings, dim=1)
        normed_array = tf.nn.l2_normalize(em2, dim=1)

        cosine_similarity = tf.matmul(normed_array, tf.transpose(normed_embeddings, [1,0]))

        init = tf.global_variables_initializer()

        with tf.Session(graph=graph) as session:
            init.run()
            log.info("Initialized\n")
            flog.write("Initialized\n")
            start = time.time()

            feed_dict = {allEmbeddings: em, em2: batchHeads}
            cosMatTailPredictions = session.run(cosine_similarity, feed_dict=feed_dict)
            feed_dict = {allEmbeddings: em, em2: batchTails}
            cosMatHeadPredictions = session.run(cosine_similarity, feed_dict=feed_dict)

            end = timeit.default_timer()
            print ("Time to evaluate model = %ds" % (end-begin) )
            flog.write ("Time to evaluate model = %ds\n" % (end-begin) )
    elif evalMethod == "l1": # L1

        allEmbeddings = tf.placeholder(tf.float32, [N,dim])
        row = tf.placeholder(tf.float32, [1,dim])
        absDiff = tf.abs(tf.subtract(row, allEmbeddings))
        l1 = tf.reduce_sum(absDiff, axis = 1)
        l1diffs = []
        session = tf.Session(graph=graph)
        # This is only tail predictions (L1 difference between head's of test set with all entities)
        for b in batchHeads:
            x = np.reshape(b, (1,dim))
            feed_dict = {allEmbeddings:em, row:x}
            temp = session.run(l1, feed_dict = feed_dict)
            result = np.reshape(temp, (1, N))
            l1diffs.append(result[0])
        cosMatTailPredictions = l1diffs
        l1diffs = []
        for b in batchTails:
            x = np.reshape(b, (1,dim))
            feed_dict = {allEmbeddings:em, row:x}
            temp = session.run(l1, feed_dict = feed_dict)
            #print ("temp shape : ", temp.shape)
            result = np.reshape(temp, (1, N))
            l1diffs.append(result[0])
        cosMatHeadPredictions = l1diffs
    else:
        print ("Unsupported evaluation method")
        sys.exit()


# Num Rows in cosine matrix must be equal to number of test triples
if (len(cosMatTailPredictions) != len(test)):
    print ("FATAL PROBLEM")
    sys.exit()

log.info("Length check passed %d" % (len(cosMatTailPredictions)))
flog.write("Length check passed %d" % (len(cosMatTailPredictions)))

def writePredictions(cosMat, test, headPredictions, evalMethod): # pass matrix of tail/head predictions, second parameter is about head-predictions or tail-predictions
    out = []
    ranks = []
    hits = 0
    for i, triple in enumerate(test):
        #log.info("Tuple(%d) - (%d, %d, %d) : " % (i, triple[0], triple[1], triple[2]))
        #flog.write("Tuple(%d) - (%d, %d, %d) : " % (i, triple[0], triple[1], triple[2]))
        array = cosMat[i]
        head = triple[0]
        tail = triple[1]
        relation = triple[2]

        if filter is not None:
            if headPredictions:
                selectedEntities = num_related_heads(relation, kbGraph)
            else:
                selectedEntities = num_related_tails(relation, kbGraph)
            array = [array[i] for i in selectedEntities]

        # Here the array is modified and does not contain N elements anymore
        # So array[34] may not be the scores of entity id 34

        #sortid = argsort(array)[::-1] # It gives the indexesof highest to lowest values in array "array"
        sortid = argsort(array)
        if headPredictions:
            if head not in selectedEntities:
                #
                rank = N
            else:
                if filter:
                    for i,s in enumerate(sortid):
                        if selectedEntities[s] == head:
                            rank = i + 1
                            break
                else:
                    rank = np.where(sortid == head)[0][0] + 1
        else:
            if tail not in sortid:
                rank = N
            else:
                if filter:
                    for i,s in enumerate(sortid):
                        if selectedEntities[s] == tail:
                            rank = i + 1
                            break
                else:
                    rank = np.where(sortid == tail)[0][0] + 1
        out.append((head, tail, rank))
        ranks.append(rank)
        if (rank < TOPK):
            hits += 1
        k = rank

        # start : result of dictionary based ranks are flawed because python dictionaries are not ordered
       # cos_dict = ddict()
       # for j,a in enumerate(array):
       #     cos_dict[j] = a
       # if evalMethod == "cosine":
       #     sorted_dict = sorted(cos_dict.items(), key = operator.itemgetter(1), reverse=True)
       # else:
       #     sorted_dict = sorted(cos_dict.items(), key = operator.itemgetter(1))

       # log.info("%d cosine results sorted, " % (len(cos_dict)))
       # flog.write("%d cosine results sorted, " % (len(cos_dict)))
       # found = False
       # for k,v in enumerate(sorted_dict):
       #     if k == TOPK:
       #         break
       #     if headPredictions:
       #         if v[0] == head:
       #             out.append((head, tail, k))
       #             found = True
       #             break
       #     else:
       #         if v[0] == tail:
       #             out.append((head, tail, k))
       #             found = True
       #             break
       # if k == TOPK:
       #     out.append((head, tail, -1))
        # end
        #log.info("Position found : %d\n" % (k))
        #flog.write("Position found : %d\n" % (k))

    prediction = "Head" if headPredictions else "Tail"

    hitRate = float(hits) / float(len(test)) * 100
    meanRank = np.mean(ranks)

    log.info("%s : Hit@%d = %f, Mean Rank = %f" % (prediction, TOPK, hitRate, meanRank))
    flog.write("%s : Hit@%d = %f, Mean Rank = %f" % (prediction, TOPK, hitRate, meanRank))
    outFile = args.fin + "-" + "TOP-" + str(TOPK) + "-" + evalMethod + "-"+ prediction+".eval.out"
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

writePredictions(cosMatTailPredictions, test, False, evalMethod)
writePredictions(cosMatHeadPredictions, test, True, evalMethod)
