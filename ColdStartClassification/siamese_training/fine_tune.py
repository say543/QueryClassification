import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import argparse

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
EPOCHS = 8
BATCH_SIZE = 4096

TRAINSET = '../data/110k_fullset_train.tsv'
TESTSET = '../data/110k_fullset_test.tsv'
THRESH = 2  # Acceptable Euclidean distance for non-matched pairs. Max=2 for unit vectors.
ALPHA = 0.001

#######################################################################################

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url, trainable=True)
print (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

#####################################################################################

# Determine epoch size
trainlength = sum(1 for line in open(TRAINSET)) - 1
epoch_size = int(np.ceil(trainlength / BATCH_SIZE))

# Load train set
train = tf.contrib.data.make_csv_dataset(TRAINSET, batch_size=BATCH_SIZE, field_delim='\t')
iter = train.make_initializable_iterator()
first = iter.get_next()
text_1, target_1 = first['text'], first['target']
second = iter.get_next()
text_2, target_2 = second['text'], second['target']
similarity = tf.cast(tf.equal(target_1, target_2), tf.float32)

# Load test set
testlength = sum(1 for line in open(TESTSET)) - 1
test_size = int(np.ceil(testlength / BATCH_SIZE))
test = tf.contrib.data.make_csv_dataset(TESTSET, batch_size=BATCH_SIZE, field_delim='\t')
test_iter = test.make_one_shot_iterator()
test_first = test_iter.get_next()
test_text_1, test_target_1 = test_first['text'], test_first['target']
test_second = test_iter.get_next()
test_text_2, test_target_2 = test_second['text'], test_second['target']
test_similarity = tf.cast(tf.equal(target_1, target_2), tf.float32)

#####################################################################################

def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")
  plt.show()
  
#####################################################################################

# Set up tensorflow
with tf.Session() as session:
  # Initialize data loader
  session.run(iter.initializer)

  embed_1 = embed(text_1)
  embed_2 = embed(text_2)
  euclidean_squared = tf.reduce_sum(tf.square(embed_1 - embed_2), 1)
  loss = (similarity)*euclidean_squared + (1-similarity)*tf.maximum(0.0,THRESH**2-euclidean_squared)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=ALPHA).minimize(loss)

  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())

  # Save initial state
  orig_text, orig_target = session.run([text_1, target_1])
  orig_pairs = zip(orig_text, orig_target)
  orig_pairs = sorted(orig_pairs, key=lambda x: x[1])
  orig_text = list(zip(*orig_pairs))[0]
  orig_embed = session.run(embed(orig_text))

  # Train
  for n in range(EPOCHS):
    # Print test loss periodically
    if n % 2 == 0:
        test_loss = 0.0
        for _ in range(test_size):
            t1, t2, tsim = session.run([test_text_1, test_text_2, test_similarity])
            batch_test_loss = session.run(loss, feed_dict={text_1: t1, text_2: t2, similarity: tsim})
            test_loss += np.mean(batch_test_loss)
        print ('** Test loss: ', test_loss/test_size)
    
    # Train in each epoch
    epoch_loss = 0.0
    for _ in range(epoch_size):
        batch_loss, _ = session.run([loss, optimizer])
        epoch_loss += np.mean(batch_loss)
    print ('Epoch ' + str(n) + ', Average Loss: ', epoch_loss/epoch_size)

  print ('Retrained... showing original embeddings, then new embeddings')

  # Show original embedding
  plot_similarity(orig_text, orig_embed, 90)
  
  # Show new embedding
  new_embed = session.run(embed(orig_text))
  plot_similarity(orig_text, new_embed, 90)

  # Save new trained model
  saver = tf.train.Saver()
  saver.save(session, '../models/fine-tuned')
