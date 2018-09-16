import ast
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import decomposition
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from classifier import classify
from reporting import accuracy, domain_classification_pra, domain_specific_pr
import argparse
from sklearn.metrics import confusion_matrix

GOLDENS = '../data/golden_queries_preprocessed.tsv'

BATCHES = 20

DOMAIN_MAP = {'MEDIACONTROL': 'media control', 'HOMEAUTOMATION': 'home automation', 'WEBNAVIGATION': 'web navigation', 'REMINDER': 'reminder', 'WEATHER': 'weather', 'CAPTURE': 'capture', 'OTHER': 'other', 'ALARM': 'alarm', 'COMMUNICATION': 'communication', 'COMMON': 'common', 'ONDEVICE': 'on device', 'OTHERSKILL': 'other skill', 'NOTE': 'note', 'CALENDAR': 'calendar', 'EMAIL': 'email', 'MYSTUFF': 'my stuff', 'PLACES': 'places'}

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

# Choose domain for P/R reporting
metric_domain = 'media control'

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', help='Path to "clean" dataset', required=True)
parser.add_argument('-m', help='Distance metric choice', choices=['cosine', 'angular', 'euclidean'], required=True)
parser.add_argument('-c', help='Classification method from embedding', choices=['argmax', 'threshold', 'prlambdas'], required=True)
parser.add_argument('-d', help='Flag for displaying sample in 3D', action='store_true')
parser.add_argument('-f', help='Flag for using a saved and fine-tuned model.', action='store_true')
parser.add_argument('-g', help='Flag for using a golden queries.', action='store_true')
parser.add_argument('-t', help='Flag for using hierarchical classification.', action='store_true')
args = parser.parse_args()

# Parse command line arguments
datapath = args.p
metric_name = args.m
classifier = args.c
display = args.d
saved_prefix = datapath[:-4] + '_'
fine_tuned = args.f
golden = args.g
hierarchical = args.t

if fine_tuned:
  saved_prefix += 'finetuned_'

if golden:
  saved_prefix += 'golden_'

if datapath[-9:-4] != 'clean':
    print ('Please choose preprocessed data file')
    exit()

# Check if embeddings already saved for this dataset
try:
  np_embed = np.load(saved_prefix + 'saved_embeddings.npz')
  np_map = np.load(saved_prefix + 'saved_mapping.npz')
  np_queries = np_embed['query']
  np_di = np_embed['di']
  queries = np_map['query'].tolist()
  targets = np_map['target'].tolist()
  candidates = np_map['candidates'].tolist()
  gq_map = np_map['gq_map'].tolist()
  domain_map = np_map['domain_map'].tolist()
  domain_rep_map = np_map['domain_rep_map'].tolist()
  domain_di_masks = np_map['domain_di_masks'].tolist()
  domains = np_map['domains'].tolist()
  print ('\nUsing saved embeddings')
  if fine_tuned:
    print ('\nCAUTION: May be using an old model.\n')
except:
  print ('\nCalculating embeddings')
  #########################################################################################
  
  # Grab data, piece together domain/intent targets
  dataset = pd.read_csv(datapath, sep='\t')

  # Separate out data
  queries = dataset.text.tolist()
  targets = dataset.target.tolist()
  di_columns = dataset[['domain', 'intent', 'target']].drop_duplicates()
  candidates = di_columns.target.tolist()
  gq_map = {k:k for k in candidates}
  candidates = list(gq_map.keys()) # In case of duplicates

  # Load in golden queries to replace default domain_intent strings
  if golden: 
    # Load golden queries from preprocessed tsv
    goldens = pd.read_csv(GOLDENS, sep='\t')
    for di_row in di_columns.iterrows(): # Make sure to only grab ones relevant to this dataset
      di_row = di_row[1]
      relevant = goldens[goldens.domain == di_row['domain']][goldens.intent == di_row['intent']]
      relevant_gqs = relevant['golden_queries']
      try:
          relevant_gqs = ast.literal_eval(relevant_gqs.item())
          for gq in relevant_gqs:
              gq_map[gq] = di_row['target']
      except: # If there are no corresponding golden queries, make an entry of the default DI string
          gq_map[di_row['target']] = di_row['target']
    candidates = list(gq_map.keys())

  # Separate domain information for hierarchical classification
  domain_map = {k:v[:v.index('.')] for k,v in gq_map.items()}
  domains = set(domain_map.values())
  domain_per_rep = np.array(list(domain_map.values()))
  domain_di_masks = {d:(domain_per_rep == d) for d in domains}
  domain_rep_map = {}
  for k, v in gq_map.items():
      domain_rep_map.setdefault(v[:v.index('.')], []).append(k)

  print ('Loaded dataset')
  
  #######################################################################################
  # Set up model, either fine tuned or tf-published

  # Set up tensorflow
  with tf.Session() as session:
    if fine_tuned:
      # Load fine tuned model
      loader = tf.train.import_meta_graph('../models/fine-tuned.meta')
      graph = tf.get_default_graph()
      tables_init = graph.get_operation_by_name('init_all_tables')
    else:
      # Import the Universal Sentence Encoder's TF Hub module
      embed = hub.Module(module_url, trainable=True)
      tables_init = tf.tables_initializer()

    # Print total params
    print (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)
  
  #####################################################################################
  # Run model to get embeddings

    session.run(tf.global_variables_initializer())
    session.run(tables_init)
    if fine_tuned:
      loader.restore(session, tf.train.latest_checkpoint('../models/'))
  
    # Embed all queries
    query_embeddings = []
    bsize = int(len(queries)/BATCHES)
    if len(queries) % BATCHES != 0:
      bsize += 1
    for n in range(BATCHES):
      print ('Batch ' + str(n))
      start = n*bsize
      end = (n+1)*bsize
      if (end > len(queries)):
        end = len(queries)
      batch = queries[start:end]
      if fine_tuned:
        in_tensor = graph.get_tensor_by_name('module/fed_input_values:0')
        out_tensor = graph.get_tensor_by_name('module/Encoder_en/hidden_layers/l2_normalize:0')
        embedded = session.run(out_tensor, feed_dict={in_tensor: batch})
      else:
        embedded = session.run(embed(batch))
      query_embeddings.append(embedded)
    query_embeddings = np.concatenate(query_embeddings)
      
    # Embed all domain/intents
    if fine_tuned:
      di_embeddings = session.run(out_tensor, feed_dict={in_tensor: candidates})
    else:
      di_embeddings = session.run(embed(candidates))

    # Save results
    np_queries = np.array(query_embeddings) 
    np_di = np.array(di_embeddings) 
    np.savez(saved_prefix + 'saved_embeddings', query=np_queries, di=np_di)
    np.savez(saved_prefix + 'saved_mapping', query=queries, target=targets, candidates=candidates, gq_map=gq_map, domain_map=domain_map, domains=domains, domain_di_masks=domain_di_masks, domain_rep_map=domain_rep_map)
    print ('Saved all embeddings')

##################################################################################
# Given all embeddings, now classify domains and intents

# Calculate means and stdev of example "music play" queries and the DI
try:
    chosen_di = 'media control play media'
    di_index = candidates.index(chosen_di)
    di_embedding = np_di[di_index]

    chosen_query_indexes = [t == chosen_di for t in targets]
    chosen_embeddings = np_queries[chosen_query_indexes]
    stdev = np.std(chosen_embeddings, axis=0)
    mean = np.mean(chosen_embeddings, axis=0)

    print ('Standard deviation of ', chosen_di, ' queries: ', np.sum(np.abs(stdev)))
    print ('Distance of DI from the query mean: ', np.sum(np.abs(di_embedding-mean)))
except:
    print ()

##################################################################################

# Visualize data
if display:
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111, projection='3d')

    # PCA to 3 dimensions
    pca = decomposition.PCA(n_components=3)
    threedim = pca.fit_transform(np_queries)
    dis = pca.transform(np_di)

    # Scatter
    ax1.scatter(threedim[:300,0], threedim[:300,1], threedim[:300,2], c='b')
    ax1.scatter(dis[:,0], dis[:,1], dis[:,2], c='r')

    ###########
    # Visualize home automation vs music
    CUTOFF = 1000
    CLASS1 = 'ONDEVICE'
    CLASS2 = 'HOMEAUTOMATION'

    ha_indexes = [DOMAIN_MAP[CLASS1] in t for t in targets] 
    music_indexes = [DOMAIN_MAP[CLASS2] in t for t in targets] 
    ha_queries = threedim[ha_indexes]
    music_queries = threedim[music_indexes]
    ax2.scatter(ha_queries[:CUTOFF,0], ha_queries[:CUTOFF,1], ha_queries[:CUTOFF,2], c='b', label=CLASS1)
    ax2.scatter(music_queries[:CUTOFF,0], music_queries[:CUTOFF,1], music_queries[:CUTOFF,2], c='r', label=CLASS2)
    fig2.legend(loc='upper left')

    plt.show()

##################################################################################

# Split into hierarchical domain / intent classification
if hierarchical:
    domain_buckets = {d: np.zeros(512) for d in domains}
    domain_counts = {d: 0 for d in domains}
    for c,e in zip(candidates, np_di):
        dom = domain_map[c]
        domain_buckets[dom] += e
        domain_counts[dom] += 1
    for d, count in domain_counts.items():
        domain_buckets[d] /= float(count)
    
    domain_embed = list(domain_buckets.values())
    domain_names = list(domain_buckets.keys())
    selected_domain, _ = classify(np_queries, domain_embed, domain_names, None, metric_name, classifier)

    # Domain classification accuracy
    dom_acc = [x[:x.index('.')] == y for x,y in zip(targets, selected_domain)]
    print ('Domain classification accuracy:', sum(dom_acc)/len(dom_acc))

    predictions = []

    # Run intent classification for the chosen domain
    intent_pools = np.array([domain_di_masks[d] for d in selected_domain])
    predictions, representative_choices = classify(np_queries, np_di, candidates, gq_map, metric_name, 'argmax', intent_pools=intent_pools)

# Classify both at once if not hierarchical
else:
    predictions, representative_choices = classify(np_queries, np_di, candidates, gq_map, metric_name, classifier)

###############################################################################
# Calculate acuracy and save predictions

targets = np.array(targets)

# Domain specific numbers
_dp, _dr, _ip, _ir = domain_specific_pr(predictions, targets, metric_domain)
print (metric_domain, 'Domain Precision: ', _dp, '  Domain Recall: ', _dr)
print (metric_domain, 'Intent Precision: ', _ip, '  Intent Recall: ', _ir)

# Get domain specific P/R
da, dp, dr = domain_classification_pra(predictions, targets)
print ('\nOverall Domain Accuracy = ', da)
print ('Overall Domain Precision: ', dp, '  Domain Recall: ', dr)

acc, acc_list = accuracy(predictions, targets)
print ('Accuracy = ', acc)

# Get precision / recall
picked_up = predictions != 'other. other'
in_domain = targets != 'other. other'
precision,_ = accuracy(predictions[picked_up], targets[picked_up])
recall,_ = accuracy(predictions[in_domain], targets[in_domain])
print ('Precision: ', precision, '  Recall: ', recall)

# Send predictions to csv
df = pd.DataFrame([queries,targets.tolist(),predictions.tolist(),representative_choices.tolist(),acc_list]).transpose()
print (df.head(n=10))
df.to_csv('../data/predictions.csv', index=False)
