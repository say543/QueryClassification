import argparse
import ast
import numpy as np
import os
import pandas as pd
import re
import sys
import tensorflow as tf
import tensorflow_hub as hub

GOLDENS = '../data/golden_queries_preprocessed.tsv'
ADDITIONAL = 'additional_domains.cfg'
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

parser = argparse.ArgumentParser()
parser.add_argument('-f', help='Flag for using a saved and fine-tuned model.', action='store_true')
args = parser.parse_args()

# Parse command line arguments
fine_tuned = args.f

# Suppress TF debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#########################################################################################

print ('\nLoading in golden queries and starting Tensorflow session...')

# Load in golden queries to replace default domain_intent strings
goldens = pd.read_csv(GOLDENS, sep='\t')

# Load in any additional domains
additional = pd.read_csv(ADDITIONAL, sep=':')
num_add = len(additional)
goldens = goldens.append(additional, ignore_index=True)

# Preprocess domain/intents
goldens = goldens[goldens.domain != 'OTHER']
goldens['target'] = goldens['domain'] + ' ' + goldens['intent']

# Print freshly added domain and intent names
added = goldens.tail(num_add)
print ('\nAdded', num_add, 'new intents for this demo, using only data from', ADDITIONAL)
print (added.target.values)
print ()

candidates = []
gq_map = {}

for golden_row in goldens.iterrows():
  golden_row = golden_row[1]
  try:
      golden_list = ast.literal_eval(golden_row['golden_queries'])
      for gq in golden_list:
        gq_map[gq] = golden_row['target']
        candidates.append(gq)
  except BaseException as e:
      print (e)
      print (golden_row)
      candidates = candidates

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

  # Reduce logging output.
  tf.logging.set_verbosity(tf.logging.ERROR)

#####################################################################################
# Run model to get embeddings

  session.run(tf.global_variables_initializer())
  session.run(tables_init)

  if fine_tuned:
    loader.restore(session, tf.train.latest_checkpoint('../models/'))

  def get_embed(inputs):
      if fine_tuned:
        embeddings = session.run(out_tensor, feed_dict={in_tensor: inputs})
      else:
        embeddings = session.run(embed(inputs))
      return embeddings

  # Embed all domain/intents
  di_embeddings = get_embed(candidates)

##################################################################################
# Given all domain_intent embeddings, now ask for a query from user

  while (True):
    print ('\nPlease enter a query: ')
    query = [sys.stdin.readline()[:-1]] # Remove newline from end
    if query == ['']:
        continue
    query_embed = get_embed(query)

    # Calculate cosine similarity
    metric = np.inner(query_embed, di_embeddings)

    # argmax
    choices = np.array(np.argmax(metric, axis=1))
    closest_goldens = np.array([candidates[x] for x in choices])
    predictions = [gq_map[x] for x in closest_goldens]
    print ('\nQuery classified as:', predictions[0])
    #print ('Closest golden:', closest_goldens[0])
    print ('---')
