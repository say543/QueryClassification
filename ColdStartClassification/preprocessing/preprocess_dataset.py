import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

# Take a dataset with query, JudgedTaskunderstandingIntent, and judged_domain
# Simplify to necessary columns, train test split
# Save cleaned file and train/test files

DOMAIN_MAP = {'MEDIACONTROL': 'media control', 'HOMEAUTOMATION': 'home automation', 'WEBNAVIGATION': 'web navigation', 'REMINDER': 'reminder', 'WEATHER': 'weather', 'CAPTURE': 'capture', 'OTHER': 'other', 'ALARM': 'alarm', 'COMMUNICATION': 'communication', 'COMMON': 'common', 'ONDEVICE': 'on device', 'OTHERSKILL': 'other skill', 'NOTE': 'note', 'CALENDAR': 'calendar', 'EMAIL': 'email', 'MYSTUFF': 'my stuff', 'PLACES': 'places'}

parser = argparse.ArgumentParser()
parser.add_argument('-p', help='Path to raw dataset', required=True)
args = parser.parse_args()

# Parse command line arguments
datapath = args.p

# Grab data, piece together domain/intent targets
dataset = pd.read_csv(datapath, sep='\t')
dataset['intent'] = dataset['JudgedTaskunderstandingIntent']
dataset['JudgedTaskunderstandingIntent'] = dataset['JudgedTaskunderstandingIntent'].str.replace('_', ' ').str.lower()
dataset['domain'] = dataset['judged_domain']
dataset['judged_domain'] = [DOMAIN_MAP[t] for t in dataset['judged_domain']]
dataset['target'] = dataset['judged_domain'] + '. ' + dataset['JudgedTaskunderstandingIntent']
dataset = dataset[['MessageText', 'domain', 'intent', 'target']]
dataset.columns = ['text', 'domain', 'intent', 'target']

# Add in data points of just DI pairs, so the siamese net will move those apart
duplicate = dataset.copy()
duplicate['text'] = duplicate['target']
joined = pd.concat([dataset, duplicate], ignore_index=True)

# Split into train and test 80/20
train, test = train_test_split(joined, test_size=0.2)

trainPath = datapath[:-4] + '_train.tsv'
testPath = datapath[:-4] + '_test.tsv'
cleanPath = datapath[:-4] + '_clean.tsv'
train.to_csv(trainPath, sep='\t', index=False)
test.to_csv(testPath, sep='\t', index=False)
dataset.to_csv(cleanPath, sep='\t', index=False)
print ('Saved preprocessed dataset to train and test splits, included cleaned file for prediction', cleanPath)
